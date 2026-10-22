
import React, { useEffect, useMemo, useRef } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { OrbitControls, Stars, Text } from "@react-three/drei";
import * as THREE from "three";

const clamp = THREE.MathUtils.clamp;
const rad = Math.PI / 180;
const val = (x, fallback = 0) =>
  Number.isFinite(Number(x)) ? Number(x) : fallback;

function toRocketRotation(q) {
  if (!Array.isArray(q) || q.length !== 4) {
    return new THREE.Quaternion();
  }

  const body = new THREE.Quaternion(
    val(q[1]), val(q[2]), val(q[3]), val(q[0], 1)
  ).normalize();

  // Physics Z-up -> Three.js Y-up.
  const frame = new THREE.Quaternion()
    .setFromRotationMatrix(
      new THREE.Matrix4().makeBasis(
        new THREE.Vector3(1, 0, 0),
        new THREE.Vector3(0, 0, -1),
        new THREE.Vector3(0, 1, 0)
      )
    );

  // Mesh +Y corresponds to physics body +Z.
  const mesh = new THREE.Quaternion()
    .setFromAxisAngle(
      new THREE.Vector3(1, 0, 0),
      Math.PI / 2
    );

  return frame.multiply(body).multiply(mesh);
}

function Logo() {
  return (
    <group position={[0, 4, 1.3]}>
      <mesh>
        <torusGeometry args={[0.52, 0.08, 12, 36]} />
        <meshBasicMaterial color="white" />
      </mesh>
      <mesh rotation={[0, 0, -Math.PI / 4]}>
        <boxGeometry args={[0.9, 0.1, 0.1]} />
        <meshBasicMaterial color="white" />
      </mesh>
    </group>
  );
}

function Flap({ position, value = 0, side = 1 }) {
  const ref = useRef();

  useFrame((_, dt) => {
    if (ref.current) {
      ref.current.rotation.y = THREE.MathUtils.damp(
        ref.current.rotation.y,
        val(value) * rad * side,
        8,
        dt
      );
    }
  });

  return (
    <group ref={ref} position={position}>
      <mesh position={[side * 0.6, 0, 0]}>
        <boxGeometry args={[1.3, 3, 0.15]} />
        <meshStandardMaterial
          color="#aab8c6"
          metalness={0.85}
          roughness={0.25}
          side={THREE.DoubleSide}
        />
      </mesh>
    </group>
  );
}

function Brake({ position, value = 0, rotation = [0, 0, 0] }) {
  const ref = useRef();

  useFrame((_, dt) => {
    if (ref.current) {
      ref.current.rotation.x = THREE.MathUtils.damp(
        ref.current.rotation.x,
        clamp(val(value), 0, 1) * 1.15,
        8,
        dt
      );
    }
  });

  return (
    <group position={position} rotation={rotation}>
      <group ref={ref}>
        <mesh position={[0, 0.6, 0]}>
          <boxGeometry args={[0.9, 1.2, 0.12]} />
          <meshStandardMaterial
            color="#8393a5"
            metalness={0.85}
            roughness={0.3}
          />
        </mesh>
      </group>
    </group>
  );
}

function Engine({ throttle, gimbal }) {
  const ref = useRef();
  const flame = useRef();

  useFrame(({ clock }, dt) => {
    if (ref.current) {
      ref.current.rotation.x = THREE.MathUtils.damp(
        ref.current.rotation.x,
        val(gimbal?.[0]) * rad,
        8,
        dt
      );

      ref.current.rotation.z = THREE.MathUtils.damp(
        ref.current.rotation.z,
        val(gimbal?.[1]) * rad,
        8,
        dt
      );
    }

    if (flame.current) {
      const t = clamp(val(throttle), 0, 1);
      flame.current.visible = t > 0.01;

      flame.current.scale.y = Math.max(
        0.001,
        t * (0.9 + 0.1 * Math.sin(clock.elapsedTime * 35))
      );
    }
  });

  return (
    <group ref={ref} position={[0, -12.5, 0]}>
      <mesh>
        <cylinderGeometry args={[0.6, 0.9, 1.5, 24]} />
        <meshStandardMaterial
          color="#2a323c"
          metalness={0.9}
          roughness={0.3}
        />
      </mesh>

      <group ref={flame}>
        <mesh position={[0, -4.6, 0]} rotation={[Math.PI, 0, 0]}>
          <coneGeometry args={[0.9, 9, 24]} />
          <meshBasicMaterial
            color="#ff8d31"
            transparent
            opacity={0.72}
            depthWrite={false}
            blending={THREE.AdditiveBlending}
          />
        </mesh>

        <mesh position={[0, -2.7, 0]} rotation={[Math.PI, 0, 0]}>
          <coneGeometry args={[0.38, 5.5, 20]} />
          <meshBasicMaterial
            color="#c7efff"
            transparent
            opacity={0.85}
            depthWrite={false}
          />
        </mesh>
      </group>
    </group>
  );
}

function Rocket({ telemetry = {} }) {
  const ref = useRef();

  const flaps = telemetry.flap_angles_deg || [0, 0, 0, 0];
  const brakes = telemetry.airbrake_deployment || [0, 0, 0, 0];

  useFrame((_, dt) => {
    if (ref.current) {
      ref.current.quaternion.slerp(
        toRocketRotation(telemetry.orientation_quat),
        1 - Math.exp(-7 * dt)
      );
    }
  });

  return (
    <group ref={ref}>
      <mesh castShadow>
        <cylinderGeometry args={[1.15, 1.25, 24, 40]} />
        <meshStandardMaterial
          color="#c0cad5"
          metalness={0.9}
          roughness={0.22}
        />
      </mesh>

      <mesh position={[0, 13.5, 0]} castShadow>
        <coneGeometry args={[1.15, 3, 40]} />
        <meshStandardMaterial
          color="#e2e8ee"
          metalness={0.85}
          roughness={0.2}
        />
      </mesh>

      <mesh position={[0, -10.5, 0]}>
        <cylinderGeometry args={[1.28, 1.28, 3, 32]} />
        <meshStandardMaterial color="#202731" />
      </mesh>

      <mesh position={[0, 4, 1.25]}>
        <planeGeometry args={[2.2, 3]} />
        <meshBasicMaterial
          color="#101821"
          side={THREE.DoubleSide}
        />
      </mesh>

      <Logo />

      <Text
        position={[0, 1.2, 1.32]}
        fontSize={0.65}
        color="#101821"
        anchorX="center"
      >
        ZERO
      </Text>

      <Flap position={[1.1, 8, 0]} value={flaps[0]} side={1} />
      <Flap position={[-1.1, 8, 0]} value={flaps[1]} side={-1} />
      <Flap position={[1.1, -8, 0]} value={flaps[2]} side={1} />
      <Flap position={[-1.1, -8, 0]} value={flaps[3]} side={-1} />

      <Brake position={[0, -4, 1.3]} value={brakes[0]} />
      <Brake
        position={[0, -4, -1.3]}
        value={brakes[1]}
        rotation={[0, Math.PI, 0]}
      />
      <Brake
        position={[1.3, -4, 0]}
        value={brakes[2]}
        rotation={[0, Math.PI / 2, 0]}
      />
      <Brake
        position={[-1.3, -4, 0]}
        value={brakes[3]}
        rotation={[0, -Math.PI / 2, 0]}
      />

      <Engine
        throttle={telemetry.throttle}
        gimbal={telemetry.gimbal_angles_deg}
      />

      <mesh
        position={[1.5, 9, 0]}
        rotation={[0, 0, -Math.PI / 2]}
        visible={Math.abs(val(telemetry.rcs_command)) > 0.05}
      >
        <coneGeometry
          args={[
            0.18,
            2 * Math.abs(val(telemetry.rcs_command)),
            12,
          ]}
        />
        <meshBasicMaterial color="#a3e6ff" />
      </mesh>
    </group>
  );
}

function LandingLights() {
  const refs = useRef([]);

  useFrame(({ clock }) => {
    refs.current.forEach((mesh, i) => {
      if (!mesh) return;

      const blink = Math.sin(
        clock.elapsedTime * 2.5 + i * 0.5
      );

      mesh.material.emissiveIntensity =
        1.8 + Math.max(0, blink) * 2.2;
    });
  });

  return (
    <group>
      {Array.from({ length: 24 }, (_, i) => {
        const angle = (i / 24) * Math.PI * 2;

        return (
          <mesh
            key={i}
            ref={(mesh) => {
              refs.current[i] = mesh;
            }}
            position={[
              Math.cos(angle) * 22,
              0.3,
              Math.sin(angle) * 22,
            ]}
          >
            <sphereGeometry args={[0.35, 10, 10]} />
            <meshStandardMaterial
              color="#b9eaff"
              emissive="#57aaff"
              emissiveIntensity={2}
            />
          </mesh>
        );
      })}
    </group>
  );
}

function LandingPad() {
  return (
    <group position={[0, -15, 0]}>
      {/* Concrete foundation */}
      <mesh position={[0, -3, 0]} receiveShadow>
        <cylinderGeometry args={[38, 42, 5, 64]} />
        <meshStandardMaterial
          color="#171e28"
          metalness={0.2}
          roughness={0.85}
        />
      </mesh>

      {/* Elevated landing deck */}
      <mesh position={[0, -0.3, 0]} receiveShadow>
        <cylinderGeometry args={[26, 27, 1.5, 64]} />
        <meshStandardMaterial
          color="#303b48"
          metalness={0.45}
          roughness={0.65}
        />
      </mesh>

      {/* Concentric guidance rings */}
      {[10, 18, 24].map((radius, i) => (
        <mesh
          key={radius}
          position={[0, 0.52 + i * 0.005, 0]}
          rotation={[-Math.PI / 2, 0, 0]}
        >
          <ringGeometry
            args={[radius - 0.14, radius + 0.14, 96]}
          />
          <meshBasicMaterial
            color={i === 2 ? "#f3b64e" : "#dbe9f8"}
            side={THREE.DoubleSide}
          />
        </mesh>
      ))}

      {/* Large landing target */}
      <mesh
        position={[0, 0.55, 0]}
        rotation={[-Math.PI / 2, 0, 0]}
      >
        <planeGeometry args={[2.8, 17]} />
        <meshBasicMaterial
          color="#f3f6fa"
          side={THREE.DoubleSide}
        />
      </mesh>

      <mesh
        position={[0, 0.56, 0]}
        rotation={[-Math.PI / 2, 0, Math.PI / 2]}
      >
        <planeGeometry args={[2.8, 17]} />
        <meshBasicMaterial
          color="#f3f6fa"
          side={THREE.DoubleSide}
        />
      </mesh>

      <LandingLights />

      {/* Surrounding ground */}
      <mesh position={[0, -5.6, 0]} receiveShadow>
        <cylinderGeometry args={[120, 120, 1, 96]} />
        <meshStandardMaterial
          color="#111821"
          roughness={0.95}
        />
      </mesh>
    </group>
  );
}

function CatchTower() {
  return (
    <group position={[-18, 0, -3]}>
      {[-2, 2].flatMap((x) =>
        [-2, 2].map((z) => (
          <mesh
            key={`${x}-${z}`}
            position={[x, 18, z]}
          >
            <boxGeometry args={[0.5, 38, 0.5]} />
            <meshStandardMaterial
              color="#8292a3"
              metalness={0.85}
              roughness={0.3}
            />
          </mesh>
        ))
      )}

      {Array.from({ length: 13 }, (_, i) => (
        <group key={i} position={[0, i * 3, 0]}>
          <mesh>
            <boxGeometry args={[5, 0.18, 5]} />
            <meshStandardMaterial color="#536374" />
          </mesh>

          <mesh rotation={[0, 0, Math.PI / 4]}>
            <boxGeometry args={[5.7, 0.12, 0.12]} />
            <meshStandardMaterial color="#9da9b6" />
          </mesh>
        </group>
      ))}

      {[-2, 2].map((z) => (
        <mesh
          key={z}
          position={[7, 22, z]}
        >
          <boxGeometry args={[14, 0.55, 0.8]} />
          <meshStandardMaterial
            color="#aebdcb"
            metalness={0.8}
            roughness={0.3}
          />
        </mesh>
      ))}

      <mesh position={[0, 39, 0]}>
        <sphereGeometry args={[0.65, 12, 12]} />
        <meshBasicMaterial color="#ffb34e" />
      </mesh>
    </group>
  );
}

function AltitudeMarkers() {
  return (
    <group position={[30, 0, 0]}>
      <mesh position={[0, 90, 0]}>
        <cylinderGeometry args={[0.07, 0.07, 180, 8]} />
        <meshBasicMaterial
          color="#6488a9"
          transparent
          opacity={0.35}
        />
      </mesh>

      {[
        0, 500, 1000, 1500, 2000,
        2500, 3000, 4000, 5000, 6000,
      ].map((height) => (
        <group
          key={height}
          position={[0, height * 0.03, 0]}
        >
          <mesh>
            <boxGeometry args={[3, 0.13, 0.13]} />
            <meshBasicMaterial color="#92c4f1" />
          </mesh>

          <Text
            position={[4, 0, 0]}
            fontSize={1.5}
            color="#d0e9ff"
            anchorX="left"
          >
            {height} m
          </Text>
        </group>
      ))}
    </group>
  );
}

function CameraDirector({ mode, rocketPosition, altitude, episode }) {
  const { camera } = useThree();

  const target = useRef(new THREE.Vector3());
  const previousMode = useRef(null);
  const previousEpisode = useRef(null);

  useFrame((_, dt) => {
    const rocket = new THREE.Vector3(...rocketPosition);

    let desiredPosition;
    let desiredTarget;

    if (mode === "launchpad") {
      // Fixed wide-angle camera.
      desiredPosition = new THREE.Vector3(115, 75, 170);
      desiredTarget = new THREE.Vector3(0, 30, 0);
    } else if (mode === "follow") {
      // Follow with visible delay.
      desiredPosition = rocket.clone().add(
        new THREE.Vector3(65, 30, 105)
      );
      desiredTarget = rocket.clone();
    } else {
      // CINEMATIC: automatically adjust zoom
      // as the rocket changes altitude.
      const zoom = clamp(
        altitude / 2500,
        0,
        1
      );

      const distance = THREE.MathUtils.lerp(
        95,
        190,
        zoom
      );

      const verticalOffset = THREE.MathUtils.lerp(
        25,
        70,
        zoom
      );

      desiredPosition = rocket.clone().add(
        new THREE.Vector3(
          distance * 0.65,
          verticalOffset,
          distance
        )
      );

      // Bias toward the landing pad during descent.
      const descent = val(altitude) < 400;

      desiredTarget = descent
        ? rocket.clone().lerp(
            new THREE.Vector3(0, -15, 0),
            0.3
          )
        : rocket.clone();
    }

    const modeChanged =
      previousMode.current !== mode;

    const episodeChanged =
      previousEpisode.current !== episode;

    if (modeChanged || episodeChanged) {
      camera.position.copy(desiredPosition);
      target.current.copy(desiredTarget);

      previousMode.current = mode;
      previousEpisode.current = episode;
    }

    // Cinematic zoom is a smooth dolly movement.
    const followSpeed =
      mode === "launchpad" ? 5 : 0.7;

    const lookSpeed =
      mode === "launchpad" ? 5 : 1.5;

    camera.position.lerp(
      desiredPosition,
      1 - Math.exp(-followSpeed * dt)
    );

    target.current.lerp(
      desiredTarget,
      1 - Math.exp(-lookSpeed * dt)
    );

    camera.lookAt(target.current);
  });

  return null;
}

function LocalScene({ telemetry, cameraMode, episode }) {
  const altitude = Math.max(
    val(telemetry.altitude_m),
    0
  );

  const p = Array.isArray(telemetry.position_m)
    ? telemetry.position_m
    : [0, 0, altitude];

  const rocketPosition = [
    val(p[0]) * 0.02,
    altitude * 0.03,
    -val(p[1]) * 0.02,
  ];

  return (
    <>
      <color attach="background" args={["#030914"]} />
      <ambientLight intensity={1.4} />
      <hemisphereLight
        args={["#94cbff", "#111821", 1.3]}
      />
      <directionalLight
        position={[65, 90, 50]}
        intensity={3}
        castShadow
      />

      <Stars
        radius={900}
        depth={200}
        count={3000}
        factor={4}
        fade
      />

      <LandingPad />
      <CatchTower />
      <AltitudeMarkers />

      <group position={rocketPosition}>
        <Rocket telemetry={telemetry} />
      </group>

      <CameraDirector
        mode={cameraMode}
        rocketPosition={rocketPosition}
        altitude={altitude}
        episode={episode}
      />
    </>
  );
}

function makeEarthTexture() {
  const canvas = document.createElement("canvas");
  canvas.width = 1024;
  canvas.height = 512;

  const ctx = canvas.getContext("2d");
  const image = ctx.createImageData(1024, 512);

  for (let y = 0; y < 512; y++) {
    const lat = (y / 512 - 0.5) * Math.PI;

    for (let x = 0; x < 1024; x++) {
      const lon = x / 1024 * Math.PI * 2;

      const terrain =
        0.55 * Math.sin(lon * 2.3 + lat) +
        0.3 * Math.sin(lon * 6.7 - lat * 4.1) +
        0.15 * Math.cos(lon * 13 + lat * 8);

      const color = Math.abs(lat) > 1.28
        ? [225, 238, 245]
        : terrain > 0.22
        ? [58, 108, 73]
        : [8, 68, 144];

      const i = (y * 1024 + x) * 4;

      image.data[i] = color[0];
      image.data[i + 1] = color[1];
      image.data[i + 2] = color[2];
      image.data[i + 3] = 255;
    }
  }

  ctx.putImageData(image, 0, 0);

  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  return texture;
}

function AnimatedEarth() {
  const surface = useRef();
  const clouds = useRef();

  const texture = useMemo(
    () => makeEarthTexture(),
    []
  );

  useEffect(() => () => texture.dispose(), [texture]);

  useFrame((_, dt) => {
    if (surface.current) {
      surface.current.rotation.y += dt * 0.012;
    }
    if (clouds.current) {
      clouds.current.rotation.y += dt * 0.017;
    }
  });

  return (
    <group>
      <mesh ref={surface}>
        <sphereGeometry args={[100, 96, 64]} />
        <meshStandardMaterial map={texture} roughness={1} />
      </mesh>

      <mesh ref={clouds}>
        <sphereGeometry args={[101.2, 80, 64]} />
        <meshStandardMaterial
          color="#d9f1ff"
          transparent
          opacity={0.12}
          depthWrite={false}
        />
      </mesh>

      <mesh>
        <sphereGeometry args={[104, 80, 64]} />
        <meshBasicMaterial
          color="#318fff"
          transparent
          opacity={0.08}
          side={THREE.BackSide}
          depthWrite={false}
        />
      </mesh>
    </group>
  );
}

function EarthScene({ telemetry }) {
  const altitude = Math.max(
    val(telemetry.altitude_m),
    0
  );

  return (
    <>
      <color attach="background" args={["#01040b"]} />
      <ambientLight intensity={0.6} />
      <directionalLight
        position={[250, 130, 180]}
        intensity={3.2}
      />
      <Stars
        radius={900}
        depth={350}
        count={4000}
        factor={4}
        fade
      />

      <AnimatedEarth />

      <group
        position={[
          104 + altitude / 6371000 * 100,
          0,
          0,
        ]}
        rotation={[0, 0, -Math.PI / 2]}
        scale={0.15}
      >
        <Rocket telemetry={telemetry} />
      </group>

      <OrbitControls
        enablePan={false}
        minDistance={130}
        maxDistance={650}
      />
    </>
  );
}

export default function RocketScene({
  telemetry,
  data,
  cameraMode = "cinematic",
  mode,
  episode = 0,
}) {
  const flight = telemetry || data || {};
  const activeMode = mode || cameraMode;
  const earthView = activeMode === "earth";

  return (
    <div
      style={{
        position: "relative",
        width: "100%",
        height: "100%",
        background: "#030914",
        overflow: "hidden",
      }}
    >
      <Canvas
        key={earthView ? "earth" : "local"}
        camera={{
          position: earthView
            ? [0, 70, 285]
            : [115, 75, 170],
          fov: 48,
          near: 0.1,
          far: 5000,
        }}
        gl={{
          antialias: true,
          alpha: false,
        }}
        dpr={[1, 1.5]}
      >
        {earthView ? (
          <EarthScene telemetry={flight} />
        ) : (
          <LocalScene
            telemetry={flight}
            cameraMode={activeMode}
            episode={episode}
          />
        )}
      </Canvas>

      <div
        style={{
          position: "absolute",
          top: 13,
          left: 13,
          padding: "9px 12px",
          borderRadius: 8,
          background: "rgba(0,0,0,.6)",
          color: "#e7f2ff",
          fontSize: 10,
          letterSpacing: 1.1,
          pointerEvents: "none",
        }}
      >
        ZERO / {activeMode.toUpperCase()} CAMERA
      </div>
    </div>
  );
}
