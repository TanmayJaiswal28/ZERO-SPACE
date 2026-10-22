
import asyncio
import math
from pathlib import Path

import numpy as np
import uvicorn

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from stable_baselines3 import PPO

from rocket.liftoff_env import LiftoffEnv


app = FastAPI(title="ZERO SPACE Liftoff Mission Control")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
)

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    ROOT
    / "models"
    / "exp004"
    / "liftoff"
    / "final_liftoff.zip"
)

TARGET_ALTITUDE = 10000.0
SIMULATION_DT = 0.5

model = None


@app.on_event("startup")
async def startup():
    global model

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}"
        )

    model = PPO.load(str(MODEL_PATH), device="cpu")

    print("ZERO SPACE LIFTOFF SERVER READY")
    print("Model:", MODEL_PATH)


@app.get("/api/status")
def status():
    return {
        "project": "ZERO SPACE",
        "experiment": "004-LIFTOFF",
        "model_loaded": model is not None,
        "target_altitude_m": TARGET_ALTITUDE,
        "control": "PPO_AUTONOMOUS",
    }


def finite(value, default=0.0):
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def build_telemetry(info, reward, episode, step):
    altitude = finite(info.get("altitude_m"))
    max_altitude = finite(
        info.get("max_altitude_m", altitude)
    )

    phase = info.get("phase", "ASCENT")

    if altitude < 1 and max_altitude < 1:
        phase = "ON PAD"

    if altitude >= TARGET_ALTITUDE:
        phase = "TARGET REACHED"

    return {
        "type": "telemetry",
        "episode": episode,
        "step": step,
        "phase": phase,
        "time_s": finite(info.get("time_s")),
        "altitude_m": altitude,
        "max_altitude_m": max_altitude,
        "speed_mps": finite(info.get("speed_mps")),
        "acceleration_mps2": finite(
            info.get("acceleration_mps2")
        ),
        "throttle": finite(info.get("throttle")),
        "fuel_kg": finite(info.get("fuel_kg")),
        "mass_kg": finite(info.get("mass_kg")),
        "dynamic_pressure_pa": finite(
            info.get("dynamic_pressure_pa")
        ),
        "heating_proxy": finite(
            info.get("heating_proxy")
        ),
        "apoapsis_m": finite(
            info.get("apoapsis_m")
        ),
        "periapsis_m": finite(
            info.get("periapsis_m")
        ),
        "eccentricity": finite(
            info.get("eccentricity")
        ),
        "inclination_deg": finite(
            info.get("inclination_deg")
        ),
        "total_reward": finite(
            info.get("total_reward")
        ),
        "reward": finite(reward),
        "success": bool(info.get("success", False)),
        "position_m": (
            np.asarray(
                info.get("position_m", [6371000, 0, 0]),
                dtype=float
            ).tolist()
        ),
    }


@app.websocket("/ws/flight")
async def flight_socket(ws: WebSocket):
    await ws.accept()

    if model is None:
        await ws.send_json({
            "type": "error",
            "message": "PPO model is not loaded",
        })
        await ws.close()
        return

    env = LiftoffEnv(
        target_altitude=TARGET_ALTITUDE
    )

    episode = 0

    try:
        await ws.send_json({
            "type": "status",
            "message": "LIFTOFF PPO AUTOPILOT ACTIVE",
        })

        while True:
            episode += 1

            observation, info = env.reset()
            total_reward = 0.0

            for step in range(1200):
                action, _ = model.predict(
                    observation,
                    deterministic=True
                )

                (
                    observation,
                    reward,
                    terminated,
                    truncated,
                    info
                ) = env.step(action)

                total_reward += float(reward)

                telemetry = build_telemetry(
                    info, reward, episode, step
                )

                telemetry["total_reward"] = total_reward

                await ws.send_json(telemetry)

                if terminated or truncated:
                    await ws.send_json({
                        "type": "episode_complete",
                        "episode": episode,
                        "success": bool(
                            info.get("success", False)
                        ),
                        "max_altitude_m": finite(
                            info.get("max_altitude_m")
                        ),
                        "total_reward": total_reward,
                    })
                    break

                # Simulation runs at approximately real time.
                await asyncio.sleep(SIMULATION_DT)

            await asyncio.sleep(3)

    except WebSocketDisconnect:
        pass

    except Exception as error:
        try:
            await ws.send_json({
                "type": "error",
                "message": str(error),
            })
        except Exception:
            pass

    finally:
        env.close()


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8004,
    )
