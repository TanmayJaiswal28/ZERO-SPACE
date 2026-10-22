
import { useEffect, useRef, useState } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import * as THREE from "three";
import "./exp003.css";

const fmt = (n, d = 1) =>
  Number.isFinite(n) ? n.toFixed(d) : "—";

function Beam({ a, b, radius = 0.035, color = "#71808d" }) {
  const start = new THREE.Vector3(...a);
  const end = new THREE.Vector3(...b);
  const direction = end.clone().sub(start);
  const midpoint = start.clone().add(end).multiplyScalar(0.5);

  const rotation = new THREE.Quaternion().setFromUnitVectors(
    new THREE.Vector3(0, 1, 0),
    direction.clone().normalize()
  );

  return (
    <mesh position={midpoint} quaternion={rotation}>
      <cylinderGeometry args={[radius, radius, direction.length(), 8]} />
      <meshStandardMaterial color={color} metalness={0.75} roughness={0.4} />
    </mesh>
  );
}

function GridFin({ side }) {
  return (
    <group position={[side * 0.54, 2.5, 0]}>
      <mesh position={[side * 0.55, 0, 0]}>
        <boxGeometry args={[1.05, 0.85, 0.08]} />
        <meshStandardMaterial color="#303b48" metalness={0.85} />
      </mesh>
      {Array.from({ length: 5 }, (_, i) => (
        <mesh key={i} position={[side * 0.55, -0.33 + i * 0.165, 0.06]}>
          <boxGeometry args={[1.02, 0.025, 0.025]} />
          <meshStandardMaterial color="#a3b2c0" metalness={0.8} />
        </mesh>
      ))}
      {Array.from({ length: 5 }, (_, i) => (
        <mesh key={i} position={[side * (0.17 + i * 0.19), 0, 0.065]}>
          <boxGeometry args={[0.025, 0.8, 0.025]} />
          <meshStandardMaterial color="#a3b2c0" metalness={0.8} />
        </mesh>
      ))}
    </group>
  );
}

function EngineCluster({ throttle }) {
  const flame = useRef();

  useFrame(({ clock }) => {
    if (!flame.current) return;
    flame.current.visible = throttle > 0.01;
    flame.current.scale.y =
      0.65 + throttle * 1.5 +
      Math.sin(clock.elapsedTime * 29) * 0.07;
  });

  return (
    <group position={[0, -3.45, 0]}>
      {Array.from({ length: 7 }, (_, i) => {
        const a = i * Math.PI * 2 / 6;
        const x = i === 0 ? 0 : Math.cos(a) * 0.3;
        const z = i === 0 ? 0 : Math.sin(a) * 0.3;
        return (
          <mesh key={i} position={[x, -0.3, z]}>
            <cylinderGeometry args={[0.09, 0.16, 0.55, 16]} />
            <meshStandardMaterial
              color="#56616c"
              metalness={0.95}
              roughness={0.22}
            />
          </mesh>
        );
      })}

      <group ref={flame} position={[0, -1.5, 0]}>
        <mesh rotation={[Math.PI, 0, 0]}>
          <coneGeometry args={[0.43, 2.5, 32]} />
          <meshBasicMaterial
            color="#ff9c39"
            transparent
            opacity={0.65}
            depthWrite={false}
          />
        </mesh>
        <mesh rotation={[Math.PI, 0, 0]}>
          <coneGeometry args={[0.22, 1.7, 24]} />
          <meshBasicMaterial color="#fff4cb" transparent opacity={0.9} />
        </mesh>
        <pointLight color="#ff9b4c" intensity={5} distance={12} />
      </group>
    </group>
  );
}

function Booster({ telemetry }) {
  const root = useRef();
  const throttle = telemetry?.throttle ?? 0;

  useFrame((_, delta) => {
    if (!root.current) return;

    const altitude = telemetry?.altitude_m ?? 0;

    // Nonlinear display scaling:
    // makes both launch and high altitude visible.
    const visualHeight =
      4.5 + 13 * Math.log1p(altitude / 1000);

    root.current.position.y = THREE.MathUtils.damp(
      root.current.position.y,
      visualHeight,
      2.5,
      delta
    );
  });

  return (
    <group ref={root} position={[0, 4.5, 0]}>
      <mesh castShadow>
        <cylinderGeometry args={[0.5, 0.5, 6.8, 64]} />
        <meshStandardMaterial
          color="#c0c8d0"
          metalness={0.88}
          roughness={0.23}
        />
      </mesh>

      {Array.from({ length: 20 }, (_, i) => (
        <mesh key={i} position={[0, -3.3 + i * 0.34, 0]}>
          <cylinderGeometry args={[0.505, 0.505, 0.018, 64]} />
          <meshStandardMaterial
            color={i % 5 === 0 ? "#596776" : "#a6b0bb"}
            metalness={0.9}
          />
        </mesh>
      ))}

      <mesh position={[0, 3.12, 0]}>
        <cylinderGeometry args={[0.51, 0.51, 0.75, 48]} />
        <meshStandardMaterial color="#161e27" metalness={0.7} />
      </mesh>

      <mesh position={[0, 3.55, 0]}>
        <coneGeometry args={[0.49, 0.45, 48]} />
        <meshStandardMaterial color="#cbd4dc" metalness={0.85} />
      </mesh>

      <GridFin side={-1} />
      <GridFin side={1} />

      {Array.from({ length: 4 }, (_, i) => {
        const a = i * Math.PI / 2;
        const x = Math.cos(a);
        const z = Math.sin(a);
        return (
          <group key={i}>
            <Beam
              a={[x * 0.45, -2.7, z * 0.45]}
              b={[x * 1.2, -4.1, z * 1.2]}
              radius={0.065}
              color="#9eacb8"
            />
            <mesh position={[x * 1.2, -4.1, z * 1.2]}>
              <boxGeometry args={[0.38, 0.12, 0.38]} />
              <meshStandardMaterial color="#566473" metalness={0.8} />
            </mesh>
          </group>
        );
      })}

      <EngineCluster throttle={throttle} />

      <mesh position={[0, 0.5, 0.51]}>
        <boxGeometry args={[0.24, 2.6, 0.018]} />
        <meshStandardMaterial color="#1a2632" />
      </mesh>
    </group>
  );
}

function Tower() {
  return (
    <group position={[-5.5, 0, 0]}>
      {[-0.8, 0.8].flatMap((x) =>
        [-0.8, 0.8].map((z) => (
          <mesh key={`${x}-${z}`} position={[x, 10, z]}>
            <boxGeometry args={[0.14, 20, 0.14]} />
            <meshStandardMaterial color="#687887" metalness={0.85} />
          </mesh>
        ))
      )}

      {Array.from({ length: 13 }, (_, i) => (
        <group key={i} position={[0, i * 1.5 + 0.7, 0]}>
          <mesh>
            <boxGeometry args={[1.9, 0.12, 1.9]} />
            <meshStandardMaterial color="#788896" metalness={0.7} />
          </mesh>
          <pointLight
            position={[0, 0.25, 0.8]}
            color="#79b8ff"
            intensity={0.3}
            distance={3}
          />
        </group>
      ))}

      {[-0.6, 0.6].map((z) => (
        <mesh key={z} position={[2.35, 9, z]}>
          <boxGeometry args={[4.8, 0.22, 0.24]} />
          <meshStandardMaterial color="#a4b3c0" metalness={0.85} />
        </mesh>
      ))}
    </group>
  );
}

function Ground() {
  return (
    <group>
      <mesh rotation={[-Math.PI / 2, 0, 0]} receiveShadow>
        <planeGeometry args={[300, 300]} />
        <meshStandardMaterial color="#1c2935" roughness={0.9} />
      </mesh>

      <gridHelper args={[200, 80, "#314557", "#253341"]} />

      <mesh position={[0, 0.16, 0]}>
        <cylinderGeometry args={[4.3, 4.3, 0.32, 96]} />
        <meshStandardMaterial color="#485561" metalness={0.45} />
      </mesh>

      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.34, 0]}>
        <ringGeometry args={[3.25, 3.35, 96]} />
        <meshBasicMaterial color="#e3a45e" side={THREE.DoubleSide} />
      </mesh>
    </group>
  );
}

function TrackingCamera({ telemetry }) {
  const target = useRef(new THREE.Vector3(0, 6, 0));

  useFrame(({ camera }, delta) => {
    const altitude = telemetry?.altitude_m ?? 0;
    const visualHeight =
      4.5 + 13 * Math.log1p(altitude / 1000);

    target.current.y = THREE.MathUtils.damp(
      target.current.y,
      visualHeight,
      1.3,
      delta
    );

    const desired = new THREE.Vector3(
      13,
      target.current.y + 5,
      24
    );

    camera.position.lerp(
      desired,
      Math.min(1, delta * 1.4)
    );

    camera.lookAt(target.current);
  });

  return null;
}

function FlightScene({ telemetry }) {
  return (
    <Canvas
      camera={{
        position: [13, 12, 24],
        fov: 42,
        near: 0.1,
        far: 500,
      }}
      shadows
      gl={{ antialias: true, preserveDrawingBuffer: true }}
    >
      <color attach="background" args={["#06111f"]} />
      <fog attach="fog" args={["#06111f", 65, 240]} />
      <ambientLight intensity={1.3} />
      <hemisphereLight args={["#a7d0ff", "#25313a", 2]} />
      <directionalLight
        position={[12, 25, 15]}
        intensity={3}
        castShadow
      />
      <directionalLight
        position={[-12, 7, -10]}
        intensity={1.5}
        color="#5f9cda"
      />

      <Ground />
      <Tower />
      <Booster telemetry={telemetry} />
      <TrackingCamera telemetry={telemetry} />
    </Canvas>
  );
}

export default function Exp003() {
  const [telemetry, setTelemetry] = useState(null);
  const [status, setStatus] = useState("CONNECTING");
  const [history, setHistory] = useState([]);

  useEffect(() => {
    let active = true;
    const ws = new WebSocket("ws://127.0.0.1:8002/ws/flight");

    ws.onopen = () => active && setStatus("CONNECTED");

    ws.onmessage = (event) => {
      if (!active) return;
      const message = JSON.parse(event.data);

      if (message.type === "telemetry") {
        setTelemetry(message);
      } else if (message.type === "status") {
        setStatus(message.message);
      } else if (message.type === "error") {
        setStatus(message.message);
      } else if (message.type === "episode_complete") {
        setHistory((prev) => [message, ...prev].slice(0, 8));
      }
    };

    ws.onerror = () => active && setStatus("CONNECTION ERROR");
    ws.onclose = () => active && setStatus((prev) =>
      prev === "CONNECTED" ? "DISCONNECTED" : prev
    );

    return () => {
      active = false;
      ws.close();
    };
  }, []);

  const metrics = [
    ["Altitude", `${fmt(telemetry?.altitude_m, 0)} m`],
    ["Vertical speed", `${fmt(telemetry?.vertical_speed_mps)} m/s`],
    ["Acceleration", `${fmt(telemetry?.acceleration_mps2)} m/s²`],
    ["Gravity", `${fmt(telemetry?.gravity_mps2, 3)} m/s²`],
    ["Fuel remaining", `${fmt(telemetry?.fuel_kg, 0)} kg`],
    ["Vehicle mass", `${fmt(telemetry?.mass_kg, 0)} kg`],
    ["Throttle", `${fmt((telemetry?.throttle ?? 0) * 100, 0)}%`],
    ["Dynamic pressure", `${fmt((telemetry?.dynamic_pressure_pa ?? 0) / 1000, 2)} kPa`],
    ["Heating indicator", `${fmt(telemetry?.heating_proxy, 0)} a.u.`],
    ["Mission time", `${fmt(telemetry?.time_s)} s`],
    ["Maximum altitude", `${fmt(telemetry?.max_altitude_m, 0)} m`],
    ["Total reward", fmt(telemetry?.total_reward)],
  ];

  const progress = Math.min(
    100,
    Math.max(0, (telemetry?.altitude_m ?? 0) / 1000)
  );

  return (
    <div className="exp003">
      <header className="exp003-header">
        <div>
          <strong>ZERO SPACE</strong>
          <span> / EXPERIMENT 003 / EARTH ASCENT</span>
        </div>
        <span className="exp003-status">{status}</span>
      </header>

      <main className="exp003-layout">
        <section className="exp003-viewport">
          <FlightScene telemetry={telemetry} />

          <div className="exp003-overlay">
            <div>MISSION / AUTONOMOUS ASCENT</div>
            <h2>{telemetry?.phase ?? "STANDBY"}</h2>
            <p>EPISODE {telemetry?.episode ?? "—"}</p>
            <p>ALTITUDE {fmt(telemetry?.altitude_m, 0)} M</p>
          </div>

          <div className="exp003-bottom">
            <span>EARTH RADIAL PHYSICS</span>
            <span>CAMERA: NONLINEAR ALTITUDE SCALE</span>
          </div>
        </section>

        <aside className="exp003-panel">
          <h3>LIVE TELEMETRY</h3>

          <div className="exp003-progress">
            <div style={{ width: `${progress}%` }} />
          </div>
          <small>100 KM ALTITUDE TARGET</small>

          {metrics.map(([label, value]) => (
            <div className="exp003-metric" key={label}>
              <span>{label}</span>
              <strong>{value}</strong>
            </div>
          ))}

          <h3>FLIGHT HISTORY</h3>
          {history.length === 0 && (
            <p className="exp003-muted">No completed flights yet</p>
          )}
          {history.map((item) => (
            <div className="exp003-metric" key={item.episode}>
              <span>Episode {item.episode}</span>
              <strong>{item.success ? "TARGET REACHED" : "FAILED"}</strong>
            </div>
          ))}

          <p className="exp003-muted">
            Educational Earth-ascent model. The displayed rocket,
            grid fins, and tower are visual assets. No orbital
            guidance, realistic reentry, or catcher contact is
            simulated in this version.
          </p>
        </aside>
      </main>
    </div>
  );
}
