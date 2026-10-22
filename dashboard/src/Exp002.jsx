
import { useEffect, useState } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import * as THREE from "three";
import "./exp002.css";

function Booster({ data }) {
  const altitude = Math.max(0, data?.altitude_m ?? 0);
  const y = altitude / 150 + 3.5;

  const x = THREE.MathUtils.clamp(
    (data?.x_m ?? 0) / 100,
    -15,
    15
  );

  const throttle = data?.throttle ?? 0;
  const active = throttle > 0.01;

  return (
    <group position={[x, y, 0]}>
      {/* Main metallic fuselage */}
      <mesh castShadow>
        <cylinderGeometry args={[0.52, 0.52, 6.5, 48]} />
        <meshStandardMaterial
          color="#bfc7cf"
          metalness={0.8}
          roughness={0.25}
        />
      </mesh>

      {/* Upper section */}
      <mesh position={[0, 3.38, 0]}>
        <coneGeometry args={[0.52, 0.6, 48]} />
        <meshStandardMaterial
          color="#e1e5e8"
          metalness={0.8}
        />
      </mesh>

      {/* Engine housing */}
      <mesh position={[0, -2.8, 0]}>
        <cylinderGeometry args={[0.55, 0.55, 0.8, 48]} />
        <meshStandardMaterial
          color="#222b34"
          metalness={0.7}
        />
      </mesh>

      {/* Control fins */}
      {[-1, 1].map((side) => (
        <group
          key={side}
          position={[side * 0.5, 2, 0]}
        >
          <mesh
            position={[side * 0.6, 0, 0]}
            rotation={[0, 0, side * 0.08]}
          >
            <boxGeometry args={[1.2, 0.7, 0.12]} />
            <meshStandardMaterial
              color="#394653"
              metalness={0.7}
            />
          </mesh>
        </group>
      ))}

      {/* Landing legs */}
      {[-1, 1].map((side) => (
        <mesh
          key={side}
          position={[side * 0.85, -3.25, 0]}
          rotation={[0, 0, side * 0.45]}
        >
          <boxGeometry args={[0.09, 1.7, 0.1]} />
          <meshStandardMaterial
            color="#9ba7b3"
            metalness={0.8}
          />
        </mesh>
      ))}

      {/* Engine plume */}
      {active && (
        <group position={[0, -4.3, 0]}>
          <mesh
            rotation={[Math.PI, 0, 0]}
            scale={[1, Math.max(0.2, throttle), 1]}
          >
            <coneGeometry args={[0.4, 2.5, 24]} />
            <meshBasicMaterial
              color="#ffb45b"
              transparent
              opacity={0.85}
            />
          </mesh>

          <pointLight
            color="#ff8b34"
            intensity={3}
            distance={7}
          />
        </group>
      )}
    </group>
  );
}

function LaunchTower() {
  return (
    <group position={[-5, 0, 0]}>
      {/* Four vertical tower supports */}
      {[0, 1, 2, 3].map((i) => (
        <mesh
          key={i}
          position={[
            i < 2 ? -0.7 : 0.7,
            9,
            i % 2 ? -0.7 : 0.7,
          ]}
        >
          <boxGeometry args={[0.12, 18, 0.12]} />
          <meshStandardMaterial
            color="#5d6a76"
            metalness={0.8}
          />
        </mesh>
      ))}

      {/* Tower platforms */}
      {Array.from({ length: 12 }, (_, i) => (
        <mesh
          key={i}
          position={[0, i * 1.5 + 0.6, 0]}
        >
          <boxGeometry args={[1.8, 0.09, 1.8]} />
          <meshStandardMaterial
            color="#71818d"
            metalness={0.7}
          />
        </mesh>
      ))}

      {/* Visual catcher arms */}
      {[-1, 1].map((side) => (
        <mesh
          key={side}
          position={[2.2, 9, side * 0.7]}
        >
          <boxGeometry args={[4.3, 0.18, 0.25]} />
          <meshStandardMaterial
            color="#b1bcc6"
            metalness={0.75}
          />
        </mesh>
      ))}
    </group>
  );
}

function FollowCamera({ data }) {
  useFrame(({ camera }, delta) => {
    const altitude =
      Math.max(0, data?.altitude_m ?? 0) / 150;

    const x = THREE.MathUtils.clamp(
      (data?.x_m ?? 0) / 100,
      -15,
      15
    );

    const target = new THREE.Vector3(
      x + 8,
      Math.max(7, altitude + 6),
      22
    );

    camera.position.lerp(
      target,
      Math.min(1, delta * 1.5)
    );

    camera.lookAt(
      x,
      Math.max(5, altitude + 3),
      0
    );
  });

  return null;
}

function Scene({ data }) {
  return (
    <Canvas
      camera={{
        position: [8, 12, 22],
        fov: 45,
        far: 400,
      }}
      shadows
      gl={{ preserveDrawingBuffer: true }}
    >
      <color
        attach="background"
        args={["#09111e"]}
      />

      <ambientLight intensity={1.7} />

      <directionalLight
        position={[10, 25, 12]}
        intensity={3}
        castShadow
      />

      {/* Ground */}
      <mesh
        rotation={[-Math.PI / 2, 0, 0]}
        receiveShadow
      >
        <planeGeometry args={[200, 200]} />
        <meshStandardMaterial color="#27333a" />
      </mesh>

      <gridHelper
        args={[120, 60, "#40505b", "#303e48"]}
      />

      {/* Launch and landing pad */}
      <mesh position={[0, 0.15, 0]}>
        <cylinderGeometry args={[5, 5, 0.3, 64]} />
        <meshStandardMaterial color="#38444d" />
      </mesh>

      <LaunchTower />
      <Booster data={data} />
      <FollowCamera data={data} />
    </Canvas>
  );
}

const num = (value, digits = 1) =>
  Number.isFinite(value)
    ? value.toFixed(digits)
    : "—";

export default function Exp002() {
  const [data, setData] = useState(null);
  const [status, setStatus] = useState("CONNECTING");
  const [events, setEvents] = useState([]);

  useEffect(() => {
    let active = true;

    const ws = new WebSocket(
      "ws://127.0.0.1:8001/ws/flight"
    );

    ws.onopen = () => {
      if (active) setStatus("CONNECTED");
    };

    ws.onmessage = (event) => {
      if (!active) return;

      const message = JSON.parse(event.data);

      if (message.type === "status") {
        setStatus(
          message.trained
            ? "PPO AUTOPILOT"
            : "DEMO CONTROLLER — PPO NOT TRAINED"
        );
      }

      if (message.type === "telemetry") {
        setData(message);
      }

      if (message.type === "episode_complete") {
        setEvents((previous) =>
          [message, ...previous].slice(0, 6)
        );
      }
    };

    ws.onerror = () => {
      if (active) setStatus("SERVER ERROR");
    };

    ws.onclose = () => {
      if (active) setStatus("DISCONNECTED");
    };

    return () => {
      active = false;
      ws.close();
    };
  }, []);

  const entries = [
    ["Altitude", num(data?.altitude_m) + " m"],
    ["Vertical speed", num(data?.vz_mps) + " m/s"],
    ["Horizontal speed", num(data?.vx_mps) + " m/s"],
    ["Total speed", num(data?.speed_mps) + " m/s"],
    [
      "Acceleration",
      num(data?.acceleration_mps2) + " m/s²",
    ],
    [
      "Fuel remaining",
      num(data?.fuel_kg, 0) + " kg",
    ],
    ["Mass", num(data?.mass_kg, 0) + " kg"],
    [
      "Thrust command",
      num((data?.throttle ?? 0) * 100, 0) + " %",
    ],
    [
      "Dynamic pressure",
      num((data?.dynamic_pressure_pa ?? 0) / 1000, 2) +
        " kPa",
    ],
    [
      "Heating proxy",
      num(data?.heat_flux_proxy, 0) + " a.u.",
    ],
    ["Horizontal offset", num(data?.x_m) + " m"],
    ["Flight time", num(data?.time_s) + " s"],
    [
      "Highest altitude",
      num(data?.max_altitude_m) + " m",
    ],
    ["Step reward", num(data?.reward, 3)],
  ];

  return (
    <div className="exp002-root">
      <header>
        <b>ZERO SPACE / EXPERIMENT 002</b>
        <span>{status}</span>
      </header>

      <div className="exp002-grid">
        <section className="exp002-scene">
          <Scene data={data} />

          <div className="exp002-overlay">
            {data?.phase ?? "READY"} · FLIGHT{" "}
            {data?.episode ?? "—"}
            <div>
              AUTONOMOUS CONTROL · NO MANUAL INPUT
            </div>
          </div>
        </section>

        <aside className="exp002-sidebar">
          <h3>FLIGHT TELEMETRY</h3>

          {entries.map(([label, value]) => (
            <div
              className="exp002-metric"
              key={label}
            >
              <span>{label}</span>
              <strong>{value}</strong>
            </div>
          ))}

          <h3>RECOVERY HISTORY</h3>

          {events.length ? (
            events.map((event) => (
              <div
                className="exp002-metric"
                key={event.episode}
              >
                <span>Episode {event.episode}</span>
                <strong>
                  {event.success
                    ? "SUCCESS"
                    : "FAILED"}
                </strong>
              </div>
            ))
          ) : (
            <p>Waiting for a completed episode</p>
          )}

          <p className="exp002-note">
            Heating is a qualitative indicator, not
            measured temperature. Catcher arms are
            visual only. This is an educational
            simulation, not flight-certified physics.
          </p>
        </aside>
      </div>
    </div>
  );
}
