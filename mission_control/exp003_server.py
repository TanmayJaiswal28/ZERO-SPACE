
import asyncio
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from stable_baselines3 import PPO

from rocket.earth_ascent_env import EarthAscentEnv


app = FastAPI(title="ZERO SPACE EXP003")

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
    "models/exp003/final_model.zip"
)


@app.get("/api/status")
def status():
    return {
        "experiment": "003",
        "model_available": MODEL_PATH.exists(),
        "target_altitude_m": 100000,
    }


@app.websocket("/ws/flight")
async def flight(ws: WebSocket):
    await ws.accept()

    if not MODEL_PATH.exists():
        await ws.send_json({
            "type": "error",
            "message": (
                "No trained model found. "
                "Run experiments.train_earth_ascent first."
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

    env = EarthAscentEnv()

    obs, info = env.reset()
    episode = 1
    total_reward = 0.0

    await ws.send_json({
        "type": "status",
        "message": "PPO AUTOPILOT ACTIVE",
    })

    try:
        while True:
            action, _ = model.predict(
                obs,
                deterministic=True,
            )

            obs, reward, terminated, truncated, info = (
                env.step(action)
            )

            total_reward += reward

            await ws.send_json({
                "type": "telemetry",
                "episode": episode,
                **info,
                "total_reward": total_reward,
            })

            if terminated or truncated:
                await ws.send_json({
                    "type": "episode_complete",
                    "episode": episode,
                    "success": info["success"],
                    "total_reward": total_reward,
                    "max_altitude_m": env.max_altitude,
                })

                await asyncio.sleep(2)

                episode += 1
                total_reward = 0.0
                obs, info = env.reset()

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
        port=8002,
    )
