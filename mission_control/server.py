
import asyncio
from pathlib import Path

import gymnasium as gym
import uvicorn

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from stable_baselines3 import PPO

from rocket.observations import RocketState


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIRECTORY = PROJECT_ROOT / "models"

AVAILABLE_MODELS = {
    "baseline",
    "time_penalty",
    "hover_reward",
    "progress_reward",
}

app = FastAPI(title="ZERO SPACE Mission Control")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/status")
def status():
    return {
        "system": "ZERO SPACE Mission Control",
        "simulation": "LunarLander-v3",
        "available_models": [
            name for name in sorted(AVAILABLE_MODELS)
            if (MODEL_DIRECTORY / name / "final_model.zip").exists()
        ],
    }


@app.websocket("/ws/flight/{model_name}")
async def flight_stream(websocket: WebSocket, model_name: str):
    await websocket.accept()

    if model_name not in AVAILABLE_MODELS:
        await websocket.send_json({"type": "error", "message": "Unknown model"})
        await websocket.close()
        return

    model_path = MODEL_DIRECTORY / model_name / "final_model.zip"

    if not model_path.exists():
        await websocket.send_json({
            "type": "error",
            "message": f"Model not found: {model_path}",
        })
        await websocket.close()
        return

    model = PPO.load(str(model_path), device="cpu")
    environment = gym.make("LunarLander-v3")

    paused = False
    restart_requested = False
    playback_speed = 1.0

    async def receive_commands():
        nonlocal paused, restart_requested, playback_speed

        while True:
            message = await websocket.receive_json()
            command = message.get("command")

            if command == "pause":
                paused = True
            elif command == "play":
                paused = False
            elif command == "restart":
                restart_requested = True
                paused = False
            elif command == "speed":
                requested = float(message.get("value", 1))
                playback_speed = max(0.25, min(4.0, requested))

    receiver = asyncio.create_task(receive_commands())

    episode = 0

    try:
        while True:
            episode += 1
            observation, _ = environment.reset()

            total_reward = 0.0
            steps = 0
            finished = False
            restart_requested = False

            while not finished:
                if restart_requested:
                    break

                if paused:
                    await asyncio.sleep(0.05)
                    continue

                action, _ = model.predict(
                    observation,
                    deterministic=True,
                )

                observation, reward, terminated, truncated, info = (
                    environment.step(int(action))
                )

                state = RocketState.from_observation(observation)

                steps += 1
                total_reward += float(reward)
                finished = terminated or truncated

                await websocket.send_json({
                    "type": "telemetry",
                    "model": model_name,
                    "episode": episode,
                    "step": steps,
                    "sim_time": steps / 50.0,
                    "position": {
                        "x": state.horizontal_position,
                        "y": state.vertical_position,
                    },
                    "velocity": {
                        "x": state.horizontal_velocity,
                        "y": state.vertical_velocity,
                    },
                    "angle": state.angle,
                    "angular_velocity": state.angular_velocity,
                    "legs": {
                        "left": state.left_leg_contact,
                        "right": state.right_leg_contact,
                    },
                    "action": int(action),
                    "reward": float(reward),
                    "total_reward": total_reward,
                    "finished": finished,
                    "terminated": terminated,
                    "truncated": truncated,
                })

                await asyncio.sleep(0.02 / playback_speed)

            if finished:
                await websocket.send_json({
                    "type": "episode_complete",
                    "episode": episode,
                    "reward": total_reward,
                    "steps": steps,
                })

                await asyncio.sleep(1)

    except (WebSocketDisconnect, RuntimeError):
        pass

    finally:
        receiver.cancel()
        environment.close()


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
    )
