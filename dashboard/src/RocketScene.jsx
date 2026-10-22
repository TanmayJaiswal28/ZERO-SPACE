
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import { useRef } from "react";
import * as THREE from "three";

const MATERIALS = {
  hull: "#b8c2cc",
  dark: "#171d25",
  steel: "#677380",
};

function CylinderBetween({ start, end, radius = 0.04, color }) {
  const from = new THREE.Vector3(...start);
  const to = new THREE.Vector3(...end);
  const direction = new THREE.Vector3().subVectors(to, from);

  const midpoint = from.clone().add(to).multiplyScalar(0.5);

  const rotation = new THREE.Quaternion().setFromUnitVectors(
    new THREE.Vector3(0, 1, 0),
    direction.clone().normalize()
  );

  return (
    <mesh position={midpoint} quaternion={rotation}>
      <cylinderGeometry args={[radius, radius, direction.length(), 10]} />
      <meshStandardMaterial
        color={color}
        metalness={0.8}
        roughness={0.32}
      />
    </mesh>
  );
}

function HullDetails() {
  return (
    <group>
      {Array.from({ length: 18 }, (_, index) => (
        <mesh key={index} position={[0, -3.4 + index * 0.39, 0]}>
          <cylinderGeometry args={[0.51, 0.51, 0.018, 48]} />
          <meshStandardMaterial
            color={index % 4 === 0 ? "#64717e" : "#a0aab5"}
            metalness={0.85}
            roughness={0.3}
          />
        </mesh>
      ))}

      {[0, Math.PI / 2, Math.PI, Math.PI * 1.5].map(
        (angle, index) => (
          <mesh
            key={index}
            position={[
              Math.sin(angle) * 0.511,
              0,
              Math.cos(angle) * 0.511,
            ]}
          >
            <boxGeometry args={[0.012, 6.8, 0.012]} />
            <meshStandardMaterial color="#74808b" />
          </mesh>
        )
      )}
    </group>
  );
}

function ControlFin({ side, action }) {
  const pivot = useRef();

  useFrame((_, delta) => {
    if (!pivot.current) return;

    const active = side === -1 ? action === 1 : action === 3;
    const targetAngle = active ? side * 0.45 : side * 0.04;

    pivot.current.rotation.y = THREE.MathUtils.damp(
      pivot.current.rotation.y,
      targetAngle,
      5,
      delta
    );
  });

  return (
    <group position={[side * 0.48, 2.45, 0]} ref={pivot}>
      <mesh>
        <cylinderGeometry args={[0.11, 0.11, 0.55, 20]} />
        <meshStandardMaterial
          color={MATERIALS.steel}
          metalness={0.95}
          roughness={0.24}
        />
      </mesh>

      <mesh position={[side * 0.65, 0, 0]}>
        <boxGeometry args={[1.25, 0.78, 0.09]} />
        <meshStandardMaterial
          color="#343d48"
          metalness={0.88}
          roughness={0.3}
        />
      </mesh>

      {[-0.24, 0, 0.24].map((offset, index) => (
        <mesh
          key={index}
          position={[side * 0.65, offset, 0.065]}
        >
          <boxGeometry args={[1.12, 0.025, 0.025]} />
          <meshStandardMaterial color="#9aa7b3" />
        </mesh>
      ))}

      <mesh position={[side * 0.2, -0.2, 0]}>
        <boxGeometry args={[0.28, 0.23, 0.22]} />
        <meshStandardMaterial color="#647383" metalness={0.9} />
      </mesh>
    </group>
  );
}

function LandingLeg({ angle }) {
  const x = Math.cos(angle);
  const z = Math.sin(angle);

  const upper = [x * 0.43, -2.3, z * 0.43];
  const lower = [x * 1.35, -4.1, z * 1.35];
  const brace = [x * 0.46, -3.15, z * 0.46];

  return (
    <group>
      <CylinderBetween
        start={upper}
        end={lower}
        radius={0.065}
        color="#a3afba"
      />
      <CylinderBetween
        start={brace}
        end={lower}
        radius={0.035}
        color="#64717f"
      />

      <mesh position={lower} rotation={[0, -angle, 0]}>
        <boxGeometry args={[0.55, 0.1, 0.4]} />
        <meshStandardMaterial
          color="#566574"
          metalness={0.8}
          roughness={0.4}
        />
      </mesh>
    </group>
  );
}

function EngineCluster({ action }) {
  const flame = useRef();
  const mainEngineActive = action === 2;

  useFrame(({ clock }) => {
    if (!flame.current) return;

    flame.current.visible = mainEngineActive;

    if (mainEngineActive) {
      const pulse = Math.sin(clock.elapsedTime * 38) * 0.08;
      flame.current.scale.y = 1 + pulse;
      flame.current.scale.x = 1 + pulse * 0.3;
      flame.current.scale.z = 1 + pulse * 0.3;
    }
  });

  return (
    <group position={[0, -3.65, 0]}>
      <mesh>
        <cylinderGeometry args={[0.48, 0.38, 0.42, 32]} />
        <meshStandardMaterial
          color="#343c46"
          metalness={0.9}
          roughness={0.35}
        />
      </mesh>

      <mesh position={[0, -0.4, 0]}>
        <cylinderGeometry args={[0.2, 0.36, 0.55, 32]} />
        <meshStandardMaterial
          color="#606b75"
          metalness={0.95}
          roughness={0.28}
        />
      </mesh>

      {Array.from({ length: 8 }, (_, index) => {
        const angle = (index / 8) * Math.PI * 2;

        return (
          <CylinderBetween
            key={index}
            start={[
              Math.cos(angle) * 0.33,
              0.15,
              Math.sin(angle) * 0.33,
            ]}
            end={[
              Math.cos(angle) * 0.24,
              -0.43,
              Math.sin(angle) * 0.24,
            ]}
            radius={0.025}
            color="#9aa5af"
          />
        );
      })}

      <group ref={flame} position={[0, -1.5, 0]}>
        <mesh rotation={[Math.PI, 0, 0]}>
          <coneGeometry args={[0.31, 2.8, 32]} />
          <meshBasicMaterial
            color="#8bbcff"
            transparent
            opacity={0.42}
            depthWrite={false}
          />
        </mesh>

        <mesh position={[0, 0.35, 0]} rotation={[Math.PI, 0, 0]}>
          <coneGeometry args={[0.16, 1.8, 24]} />
          <meshBasicMaterial
            color="#fff3d7"
            transparent
            opacity={0.92}
          />
        </mesh>

        <pointLight color="#7baeff" intensity={5} distance={8} />
      </group>
    </group>
  );
}

function ReusableBooster({ flight }) {
  const rocketRef = useRef();
  const action = flight?.action ?? 0;

  useFrame((_, delta) => {
    if (!rocketRef.current) return;

    const targetX = THREE.MathUtils.clamp(
      (flight?.position?.x ?? 0) * 2.2,
      -4,
      4
    );

    const targetY = THREE.MathUtils.clamp(
      5.2 + (flight?.position?.y ?? 0) * 2.2,
      4.4,
      9.5
    );

    rocketRef.current.position.x = THREE.MathUtils.damp(
      rocketRef.current.position.x,
      targetX,
      5,
      delta
    );

    rocketRef.current.position.y = THREE.MathUtils.damp(
      rocketRef.current.position.y,
      targetY,
      5,
      delta
    );

    rocketRef.current.rotation.z = THREE.MathUtils.damp(
      rocketRef.current.rotation.z,
      -(flight?.angle ?? 0),
      5,
      delta
    );
  });

  return (
    <group ref={rocketRef} position={[0, 8, 0]}>
      <mesh>
        <cylinderGeometry args={[0.5, 0.5, 7.2, 48]} />
        <meshStandardMaterial
          color={MATERIALS.hull}
          metalness={0.83}
          roughness={0.28}
        />
      </mesh>

      <HullDetails />

      <mesh position={[0, 3.35, 0]}>
        <cylinderGeometry args={[0.505, 0.505, 0.55, 48]} />
        <meshStandardMaterial
          color={MATERIALS.dark}
          metalness={0.65}
          roughness={0.42}
        />
      </mesh>

      <mesh position={[0, 3.68, 0]}>
        <cylinderGeometry args={[0.48, 0.5, 0.12, 48]} />
        <meshStandardMaterial color="#778491" metalness={0.8} />
      </mesh>

      <ControlFin side={-1} action={action} />
      <ControlFin side={1} action={action} />

      {[0, Math.PI / 2, Math.PI, Math.PI * 1.5].map(
        (angle, index) => (
          <LandingLeg key={index} angle={angle} />
        )
      )}

      <EngineCluster action={action} />

      <mesh position={[0, 0.6, 0.515]}>
        <boxGeometry args={[0.23, 2.5, 0.018]} />
        <meshStandardMaterial
          color="#19232d"
          metalness={0.5}
          roughness={0.45}
        />
      </mesh>
    </group>
  );
}

function LandingZone() {
  return (
    <group>
      <mesh position={[0, -0.15, 0]}>
        <cylinderGeometry args={[4, 4, 0.25, 96]} />
        <meshStandardMaterial
          color="#242c35"
          metalness={0.35}
          roughness={0.75}
        />
      </mesh>

      <mesh
        position={[0, -0.01, 0]}
        rotation={[-Math.PI / 2, 0, 0]}
      >
        <ringGeometry args={[3.2, 3.26, 96]} />
        <meshBasicMaterial
          color="#d3a168"
          side={THREE.DoubleSide}
        />
      </mesh>

      <gridHelper
        args={[100, 80, "#27313b", "#171e27"]}
        position={[0, -0.29, 0]}
      />
    </group>
  );
}

/*
 * Cinematic camera:
 * Follows the rocket while slowly orbiting around it.
 * The camera is presentation-only and does not affect physics.
 */
function CinematicCamera({ flight, enabled }) {
  const target = useRef(new THREE.Vector3(0, 5, 0));

  useFrame(({ camera, clock }, delta) => {
    if (!enabled) return;

    const rocketX = THREE.MathUtils.clamp(
      (flight?.position?.x ?? 0) * 2.2,
      -4,
      4
    );

    const rocketY = THREE.MathUtils.clamp(
      5.2 + (flight?.position?.y ?? 0) * 2.2,
      4.4,
      9.5
    );

    target.current.lerp(
      new THREE.Vector3(rocketX, rocketY, 0),
      Math.min(1, delta * 2)
    );

    const angle = clock.elapsedTime * 0.13;
    const radius = 15;

    const desiredPosition = new THREE.Vector3(
      target.current.x + Math.sin(angle) * radius,
      target.current.y + 3.5,
      Math.cos(angle) * radius
    );

    camera.position.lerp(
      desiredPosition,
      Math.min(1, delta * 1.5)
    );

    camera.lookAt(target.current);
  });

  return null;
}

export default function RocketScene({ flight, cinematic = false }) {
  return (
    <Canvas
      camera={{
        position: [13, 8, 21],
        fov: 38,
        near: 0.1,
        far: 250,
      }}
      shadows
      gl={{ antialias: true, preserveDrawingBuffer: true }}
      style={{ width: "100%", height: "100%" }}
    >
      <color attach="background" args={["#080d14"]} />
      <fog attach="fog" args={["#080d14", 35, 110]} />

      <ambientLight intensity={1.4} />

      <directionalLight
        position={[8, 15, 10]}
        intensity={3}
        color="#f0f4ff"
      />

      <directionalLight
        position={[-9, 5, -8]}
        intensity={2}
        color="#779bc7"
      />

      <spotLight
        position={[0, 18, 8]}
        angle={0.4}
        penumbra={0.6}
        intensity={3}
      />

      <ReusableBooster flight={flight} />
      <LandingZone />

      <CinematicCamera flight={flight} enabled={cinematic} />

      <OrbitControls
        enabled={!cinematic}
        enablePan={false}
        minDistance={12}
        maxDistance={45}
        target={[0, 5, 0]}
        enableDamping
        dampingFactor={0.06}
      />
    </Canvas>
  );
}
