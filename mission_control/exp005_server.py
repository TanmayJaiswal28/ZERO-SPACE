
import asyncio
import contextlib
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket
from fastapi.websockets import WebSocketDisconnect
from stable_baselines3 import PPO

from rocket.sixdof_env import SixDOFEnv


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "exp005" / "stage2" / "attitude_final.zip"

app = FastAPI(title="ZERO SPACE / EXP005")
model = None


@app.on_event("startup")
async def startup():
    global model

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Stage 2 PPO model missing: {MODEL_PATH}"
        )

    model = PPO.load(str(MODEL_PATH))
    print("ZERO SPACE EXP005 / STAGE 2 READY")
    print("WebSocket: ws://127.0.0.1:8005/ws/flight")


@app.get("/")
def status():
    return {
        "project": "ZERO SPACE",
        "experiment": "005",
        "stage": 2,
        "ready": model is not None,
        "websocket": "/ws/flight",
    }


def json_safe(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


@app.websocket("/ws/flight")
async def flight(websocket: WebSocket):
    await websocket.accept()

    if model is None:
        await websocket.send_json({
            "type": "error",
            "message": "PPO model is not loaded",
        })
        await websocket.close()
        return

    env = SixDOFEnv()

    episode = 0
    observation = None
    info = {}

    paused = False
    auto_rerun = True
    playback_speed = 1.0
    running = False
    finished = False
    reset_requested = True
    next_episode_at = None

    async def receive_commands():
        nonlocal paused, auto_rerun
        nonlocal playback_speed, reset_requested
        nonlocal next_episode_at

        try:
            while True:
                packet = await websocket.receive_json()

                command = str(
                    packet.get("command", "")
                ).lower()

                if command in ("rerun", "reset", "start"):
                    reset_requested = True
                    paused = False
                    next_episode_at = None

                elif command == "pause":
                    paused = True

                elif command == "resume":
                    paused = False

                elif command == "speed":
                    requested = float(
                        packet.get("value", 1)
                    )

                    playback_speed = min(
                        4.0,
                        max(0.5, requested)
                    )

                elif command == "auto_rerun":
                    auto_rerun = bool(
                        packet.get("enabled", True)
                    )

                await websocket.send_json({
                    "type": "control_state",
                    "paused": paused,
                    "auto_rerun": auto_rerun,
                    "playback_speed": playback_speed,
                    "reset_requested": reset_requested,
                })

        except WebSocketDisconnect:
            return

    receiver = asyncio.create_task(receive_commands())

    try:
        loop = asyncio.get_running_loop()

        while not receiver.done():
            if reset_requested:
                reset_requested = False
                episode += 1

                observation, info = env.reset(
                    seed=episode
                )

                paused = False
                running = True
                finished = False
                next_episode_at = None

                await websocket.send_json({
                    "type": "episode_started",
                    "episode": episode,
                    "message": "MISSION RUNNING",
                    "paused": False,
                    "auto_rerun": auto_rerun,
                    "playback_speed": playback_speed,
                    **json_safe(info),
                })

            if finished:
                if (
                    auto_rerun
                    and next_episode_at is not None
                    and loop.time() >= next_episode_at
                ):
                    reset_requested = True
                    continue

                await asyncio.sleep(0.05)
                continue

            if paused:
                await asyncio.sleep(0.05)
                continue

            if not running:
                await asyncio.sleep(0.05)
                continue

            action, _ = model.predict(
                observation,
                deterministic=True
            )

            observation, reward, terminated, truncated, info = (
                env.step(action)
            )

            telemetry = json_safe(info)

            # Use the environment's actual state.
            # Do not synthesize altitude or actuator readings.
            await websocket.send_json({
                **telemetry,
                "type": "telemetry",
                "experiment": "005",
                "stage": 2,
                "episode": episode,
                "controller": "PPO 6DOF AUTOPILOT",
                "paused": paused,
                "playback_speed": playback_speed,
                "auto_rerun": auto_rerun,
            })

            if terminated or truncated:
                running = False
                finished = True

                success = bool(
                    telemetry.get("success", False)
                )

                await websocket.send_json({
                    "type": "episode_complete",
                    "episode": episode,
                    "success": success,
                    "phase": telemetry.get(
                        "phase", "COMPLETE"
                    ),
                    "time_s": telemetry.get(
                        "time_s", 0
                    ),
                    "total_reward": telemetry.get(
                        "total_reward", 0
                    ),
                    "truncated": bool(truncated),
                })

                if auto_rerun:
                    next_episode_at = (
                        loop.time() + 2.0
                    )

            # Cooperative delay allows commands to be received.
            dt = max(
                0.01,
                float(getattr(env, "dt", 0.05))
            )

            await asyncio.sleep(
                dt / playback_speed
            )

    except (
        WebSocketDisconnect,
        ConnectionResetError,
        RuntimeError,
    ):
        pass

    except Exception as exc:
        with contextlib.suppress(Exception):
            await websocket.send_json({
                "type": "error",
                "message": str(exc),
            })
        raise

    finally:
        receiver.cancel()

        with contextlib.suppress(
            asyncio.CancelledError
        ):
            await receiver

        env.close()


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8005,
        log_level="info",
    )
