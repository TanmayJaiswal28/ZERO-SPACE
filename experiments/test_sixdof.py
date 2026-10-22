
import numpy as np

from rocket.sixdof_env import SixDOFEnv


def run(action, seed=123, steps=10):
    env = SixDOFEnv()
    env.reset(seed=seed)

    result = None

    for _ in range(steps):
        _, _, done, trunc, result = env.step(action)

        if done or trunc:
            break

    env.close()
    return result


def main():
    neutral = np.array([
        -1,  # Main engine off
         0,  # Gimbal X
         0,  # Gimbal Y
         0, 0, 0, 0,  # Four flaps
        -1, -1, -1, -1,  # Four airbrakes closed
         0,  # RCS off
    ], dtype=np.float32)

    baseline = run(neutral)

    assert len(baseline["orientation_quat"]) == 4
    assert len(baseline["flap_angles_deg"]) == 4
    assert len(baseline["airbrake_deployment"]) == 4
    assert np.isfinite(baseline["altitude_m"])

    print("PASS: 6DOF state")

    # Test airbrakes.
    brakes = neutral.copy()
    brakes[7:11] = 1

    b = run(brakes)

    assert all(
        x > 0.01
        for x in b["airbrake_deployment"]
    )

    assert (
        b["vertical_velocity_mps"]
        > baseline["vertical_velocity_mps"]
    ), "Airbrakes must reduce downward speed"

    print("PASS: aerodynamic airbrakes")

    # Test flap-generated torque.
    flaps = neutral.copy()
    flaps[3] = 1

    f = run(flaps)

    assert (
        abs(
            f["torque_nm"][1]
            - baseline["torque_nm"][1]
        ) > 1
    ), "Flap must change pitch torque"

    print("PASS: flap control torque")

    # Test RCS yaw torque.
    rcs = neutral.copy()
    rcs[11] = 1

    r = run(rcs)

    assert (
        r["torque_nm"][2]
        > baseline["torque_nm"][2] + 10000
    ), "RCS must create yaw torque"

    print("PASS: RCS yaw thruster")

    # Test gimballed engine.
    engine = neutral.copy()
    engine[0] = 1
    engine[1] = 1

    e = run(engine)

    assert (
        e["fuel_kg"]
        < baseline["fuel_kg"]
    ), "Engine must consume fuel"

    assert (
        abs(e["gimbal_angles_deg"][0]) > 0
    ), "Gimbal must move"

    print("PASS: fuel consumption")
    print("PASS: engine gimbal")

    print()
    print("ZERO SPACE / STAGE 2 PHYSICS TEST")
    print("-" * 50)

    print(
        "Baseline descent:",
        round(baseline["vertical_velocity_mps"], 2),
        "m/s"
    )

    print(
        "Braked descent:",
        round(b["vertical_velocity_mps"], 2),
        "m/s"
    )

    print()
    print("ALL COMPONENT TESTS PASSED")


if __name__ == "__main__":
    main()
