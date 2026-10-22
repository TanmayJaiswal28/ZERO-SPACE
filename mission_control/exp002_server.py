
"""ZERO SPACE Experiment 002 telemetry server."""

import asyncio
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from stable_baselines3 import PPO
from rocket.launch_return_env import LaunchReturnEnv


app = FastAPI(title="ZERO SPACE EXP002")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
)

MODEL = Path("models/exp002/final_model.zip")


@app.get("/api/status")
def status():
    return {
        "experiment": "002",
        "trained_model_available": MODEL.exists(),
        "model_path": str(MODEL),
    }


@app.websocket("/ws/flight")
async def flight(ws: WebSocket):
    await ws.accept()

    env = LaunchReturnEnv()

    model = (
        PPO.load(str(MODEL), device="cpu")
        if MODEL.exists()
        else None
    )

    await ws.send_json({
        "type": "status",
        "trained": model is not None,
        "message": (
            "Trained PPO policy"
            if model
            else "Demo controller: PPO not trained"
        ),
    })

    obs, info = env.reset()
    episode = 1

    try:
        while True:
            if model is not None:
                # AI automatically controls the rocket
                action, _ = model.predict(
                    obs,
                    deterministic=True,
                )

            else:
                # Automatic demo controller.
                # This is NOT the trained AI.
                if env.phase == 0:
                    throttle = (
                        0.72 if env.z < 950 else 0.0
                    )
                else:
                    mass = env.dry_mass + env.fuel

                    braking_accel = max(
                        env.max_thrust / mass - env.g,
                        1.0,
                    )

                    stopping_distance = (
                        env.vz**2 / (2 * braking_accel)
                    )

                    throttle = (
                        0.9
                        if env.vz < 0
                        and env.z < stopping_distance + 120
                        else 0.0
                    )

                lateral = max(
                    -1.0,
                    min(
                        1.0,
                        -0.004 * env.x - 0.03 * env.vx,
                    ),
                )

                action = [throttle, lateral]

            obs, reward, terminated, truncated, info = (
                env.step(action)
            )

            await ws.send_json({
                "type": "telemetry",
                "episode": episode,
                "trained": model is not None,
                **info,
                "success": info.get("success", False),
            })

            if terminated or truncated:
                await ws.send_json({
                    "type": "episode_complete",
                    "episode": episode,
                    "success": info.get("success", False),
                    "max_altitude_m": env.max_altitude,
                })

                await asyncio.sleep(2)

                episode += 1
                obs, info = env.reset()

            await asyncio.sleep(env.dt)

    except (WebSocketDisconnect, RuntimeError):
        pass


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8001,
    )
