
import asyncio
import csv
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from stable_baselines3 import PPO

from rocket.orbital_env import OrbitalEnv


app = FastAPI(title="ZERO SPACE EXP004")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = Path(
    "models/exp004/final_model.zip"
)

LOG_PATH = Path(
    "models/exp004/flight_log.csv"
)


@app.get("/api/status")
def status():
    return {
        "experiment": "004",
        "model_available": MODEL_PATH.exists(),
        "physics": "3D Earth-centered point mass",
    }


@app.websocket("/ws/flight")
async def flight(ws: WebSocket):
    await ws.accept()

    if not MODEL_PATH.exists():
        await ws.send_json({
            "type": "error",
            "message": (
                "Train the EXP004 PPO model first. "
                "Run python -m experiments.train_orbital"
            ),
        })
        await ws.close()
        return

    try:
        model = PPO.load(
            str(MODEL_PATH),
            device="cpu",
        )
    except Exception as exc:
        await ws.send_json({
            "type": "error",
            "message": f"Model load failed: {exc}",
        })
        await ws.close()
        return

    env = OrbitalEnv()
    observation, _ = env.reset()
    episode = 1

    LOG_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    columns = [
        "episode",
        "time_s",
        "altitude_m",
        "speed_mps",
        "fuel_kg",
        "dynamic_pressure_pa",
        "periapsis_m",
        "apoapsis_m",
        "phase",
        "success",
    ]

    await ws.send_json({
        "type": "status",
        "message": "EXP004 PPO AUTOPILOT ACTIVE",
    })

    try:
        with LOG_PATH.open(
            "w",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=columns
            )
            writer.writeheader()

            while True:
                action, _ = model.predict(
                    observation,
                    deterministic=True
                )

                (
                    observation,
                    reward,
                    terminated,
                    truncated,
                    info,
                ) = env.step(action)

                payload = {
                    "type": "telemetry",
                    "episode": episode,
                    **info,
                }

                await ws.send_json(payload)

                writer.writerow({
                    key: (
                        payload.get(key)
                        if payload.get(key) is not None
                        else ""
                    )
                    for key in columns
                })

                if terminated or truncated:
                    await ws.send_json({
                        "type": "episode_complete",
                        "episode": episode,
                        "success": info["success"],
                        "max_altitude_m": info["max_altitude_m"],
                        "total_reward": info["total_reward"],
                    })

                    file.flush()

                    await asyncio.sleep(2)

                    episode += 1
                    observation, _ = env.reset()

                await asyncio.sleep(env.dt)

    except WebSocketDisconnect:
        pass
    finally:
        env.close()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8003,
    )
