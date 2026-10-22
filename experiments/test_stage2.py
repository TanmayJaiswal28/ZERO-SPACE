
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from rocket.sixdof_env import SixDOFEnv


MODEL_PATH = Path(
    "models/exp005/stage2/attitude_final.zip"
)

EPISODES = 20


def get_tilt_degrees(quaternion):
    w, x, y, z = quaternion

    # Body +Z direction in world coordinates
    up_z = 1 - 2 * (x * x + y * y)

    up_z = np.clip(up_z, -1.0, 1.0)

    return float(
        np.degrees(np.arccos(up_z))
    )


def main():
    if not MODEL_PATH.exists():
        print("ERROR: Trained model not found.")
        print("Expected:", MODEL_PATH.resolve())
        return

    print("=" * 65)
    print("ZERO SPACE / EXP005 / STAGE 2")
    print("AUTONOMOUS 6DOF FLIGHT CONTROLLER TEST")
    print("=" * 65)

    model = PPO.load(str(MODEL_PATH))

    env = SixDOFEnv()

    results = []

    for episode in range(1, EPISODES + 1):
        obs, info = env.reset(seed=episode)

        initial_tilt = get_tilt_degrees(
            info["orientation_quat"]
        )

        terminated = False
        truncated = False

        max_tilt = initial_tilt

        flap_activity = []
        brake_activity = []
        gimbal_activity = []
        rcs_activity = []

        while not (terminated or truncated):
            action, _ = model.predict(
                obs,
                deterministic=True
            )

            obs, reward, terminated, truncated, info = (
                env.step(action)
            )

            tilt = get_tilt_degrees(
                info["orientation_quat"]
            )

            max_tilt = max(max_tilt, tilt)

            flap_activity.append(
                max(
                    abs(x)
                    for x in info["flap_angles_deg"]
                )
            )

            brake_activity.append(
                max(info["airbrake_deployment"])
            )

            gimbal_activity.append(
                max(
                    abs(x)
                    for x in info["gimbal_angles_deg"]
                )
            )

            rcs_activity.append(
                abs(info["rcs_command"])
            )

        final_tilt = get_tilt_degrees(
            info["orientation_quat"]
        )

        success = bool(
            info.get("success", False)
        )

        results.append({
            "success": success,
            "initial_tilt": initial_tilt,
            "final_tilt": final_tilt,
            "max_tilt": max_tilt,
            "speed": abs(
                info["vertical_velocity_mps"]
            ),
            "fuel": info["fuel_kg"],
            "time": info["time_s"],
            "flap": max(flap_activity, default=0),
            "brake": max(brake_activity, default=0),
            "gimbal": max(gimbal_activity, default=0),
            "rcs": max(rcs_activity, default=0),
        })

        status = "LANDED" if success else (
            "TIMEOUT" if truncated else "IMPACT"
        )

        print(
            f"EP {episode:02d} | "
            f"{status:7s} | "
            f"TILT {initial_tilt:5.1f} -> "
            f"{final_tilt:5.1f} deg | "
            f"VZ {abs(info['vertical_velocity_mps']):6.1f} m/s | "
            f"FUEL {info['fuel_kg']:7.0f} kg"
        )

    env.close()

    successes = sum(
        result["success"]
        for result in results
    )

    def average(key):
        return float(np.mean([
            result[key]
            for result in results
        ]))

    print()
    print("=" * 65)
    print("FINAL PERFORMANCE REPORT")
    print("=" * 65)

    print(
        f"Successful landings : "
        f"{successes}/{EPISODES}"
    )

    print(
        f"Landing success rate: "
        f"{100 * successes / EPISODES:.1f}%"
    )

    print(
        f"Average initial tilt: "
        f"{average('initial_tilt'):.2f} deg"
    )

    print(
        f"Average final tilt  : "
        f"{average('final_tilt'):.2f} deg"
    )

    print(
        f"Average maximum tilt: "
        f"{average('max_tilt'):.2f} deg"
    )

    print(
        f"Average final |Vz|  : "
        f"{average('speed'):.2f} m/s"
    )

    print(
        f"Average flight time : "
        f"{average('time'):.2f} s"
    )

    print()
    print("ACTUATOR ACTIVITY")

    print(
        f"Max flap deflection : "
        f"{max(r['flap'] for r in results):.2f} deg"
    )

    print(
        f"Max airbrake deploy : "
        f"{max(r['brake'] for r in results):.3f}"
    )

    print(
        f"Max gimbal deflect  : "
        f"{max(r['gimbal'] for r in results):.2f} deg"
    )

    print(
        f"Max RCS command     : "
        f"{max(r['rcs'] for r in results):.3f}"
    )

    print()
    print("NOTE: Actuator activity does not by itself")
    print("prove that an actuator improves control.")


if __name__ == "__main__":
    main()

 