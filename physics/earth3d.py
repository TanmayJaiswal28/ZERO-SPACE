
import numpy as np
import math

# Earth reference constants
EARTH_RADIUS = 6_371_000.0       # m
EARTH_MU = 3.986004418e14        # m^3/s^2
EARTH_ROTATION = 7.2921159e-5   # rad/s
SEA_LEVEL_DENSITY = 1.225       # kg/m^3
STANDARD_GRAVITY = 9.80665


def norm(vector):
    return float(np.linalg.norm(vector))


def unit(vector):
    length = norm(vector)
    if length < 1e-12:
        return np.zeros(3, dtype=float)
    return np.asarray(vector, dtype=float) / length


def altitude(position):
    return norm(position) - EARTH_RADIUS


def gravity(position):
    r = norm(position)

    if r < EARTH_RADIUS * 0.5:
        raise ValueError("Invalid Earth-centered position")

    return -EARTH_MU * position / r**3


def atmosphere_density(height):
    """
    Simplified exponential atmosphere.
    Not a validated thermosphere model.
    """
    h = max(0.0, height)

    return SEA_LEVEL_DENSITY * math.exp(-h / 8500.0)


def atmosphere_velocity(position):
    """Velocity of the co-rotating atmosphere."""
    omega = np.array([0.0, 0.0, EARTH_ROTATION])
    return np.cross(omega, position)


def aerodynamic_drag(position, velocity, cd, area):
    h = altitude(position)
    rho = atmosphere_density(h)

    relative_velocity = (
        velocity - atmosphere_velocity(position)
    )

    speed = norm(relative_velocity)

    force = (
        -0.5
        * rho
        * speed
        * cd
        * area
        * relative_velocity
    )

    dynamic_pressure = 0.5 * rho * speed**2

    # Qualitative heating indicator, not W/m^2.
    heating_proxy = math.sqrt(
        rho / SEA_LEVEL_DENSITY
    ) * speed**3

    return force, dynamic_pressure, heating_proxy


def orbital_elements(position, velocity):
    """
    Compute classical two-body orbit diagnostics.

    Inclination is referenced to Earth's equatorial plane.
    Apoapsis/periapsis are meaningful for bound orbits.
    """
    r = norm(position)
    v = norm(velocity)

    angular_momentum = np.cross(position, velocity)
    h = norm(angular_momentum)

    eccentricity_vector = (
        np.cross(velocity, angular_momentum) / EARTH_MU
        - position / r
    )

    eccentricity = norm(eccentricity_vector)

    energy = v**2 / 2 - EARTH_MU / r

    semi_major_axis = (
        -EARTH_MU / (2 * energy)
        if abs(energy) > 1e-9
        else float("inf")
    )

    inclination = (
        math.degrees(
            math.acos(
                np.clip(
                    angular_momentum[2] / h,
                    -1.0,
                    1.0
                )
            )
        )
        if h > 1e-9
        else 0.0
    )

    if energy < 0 and eccentricity < 1:
        periapsis = (
            semi_major_axis * (1 - eccentricity)
            - EARTH_RADIUS
        )
        apoapsis = (
            semi_major_axis * (1 + eccentricity)
            - EARTH_RADIUS
        )
    else:
        periapsis = None
        apoapsis = None

    return {
        "eccentricity": eccentricity,
        "inclination_deg": inclination,
        "specific_energy": energy,
        "semi_major_axis_m": semi_major_axis,
        "periapsis_m": periapsis,
        "apoapsis_m": apoapsis,
    }


def derivatives(position, velocity, mass, thrust, cd, area):
    drag, q, heat = aerodynamic_drag(
        position, velocity, cd, area
    )

    acceleration = (
        gravity(position)
        + (thrust + drag) / mass
    )

    return velocity, acceleration, q, heat


def integrate_rk4(
    position,
    velocity,
    mass,
    thrust,
    cd,
    area,
    dt
):
    """RK4 integration with constant force/mass over dt."""

    def f(p, v):
        dp, dv, _, _ = derivatives(
            p, v, mass, thrust, cd, area
        )
        return dp, dv

    k1p, k1v = f(position, velocity)

    k2p, k2v = f(
        position + 0.5 * dt * k1p,
        velocity + 0.5 * dt * k1v
    )

    k3p, k3v = f(
        position + 0.5 * dt * k2p,
        velocity + 0.5 * dt * k2v
    )

    k4p, k4v = f(
        position + dt * k3p,
        velocity + dt * k3v
    )

    new_position = position + (
        dt / 6
    ) * (k1p + 2*k2p + 2*k3p + k4p)

    new_velocity = velocity + (
        dt / 6
    ) * (k1v + 2*k2v + 2*k3v + k4v)

    _, acceleration, q, heat = derivatives(
        new_position,
        new_velocity,
        mass,
        thrust,
        cd,
        area
    )

    return (
        new_position,
        new_velocity,
        acceleration,
        q,
        heat
    )
