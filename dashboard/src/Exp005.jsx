
import React, { useEffect, useRef, useState } from "react";
import RocketScene from "./exp004/RocketScene.jsx";
import "./exp004.css";

const WS = "ws://127.0.0.1:8005/ws/flight";

const INITIAL = {
  altitude_m: 0,
  speed_mps: 0,
  vertical_velocity_mps: 0,
  acceleration_mps2: 0,
  dynamic_pressure_pa: 0,
  fuel_kg: 0,
  mass_kg: 0,
  throttle: 0,
  time_s: 0,
  phase: "STANDBY",
  episode: 0,
  orientation_quat: [1, 0, 0, 0],
  flap_angles_deg: [0, 0, 0, 0],
  airbrake_deployment: [0, 0, 0, 0],
  gimbal_angles_deg: [0, 0],
  rcs_command: 0,
  position_m: [0, 0, 0],
};

const n = (value, digits = 1) => {
  const valueNumber = Number(value);
  return Number.isFinite(valueNumber)
    ? valueNumber.toFixed(digits)
    : "0";
};

const clamp = (v, min, max) =>
  Math.max(min, Math.min(max, v));

const panel = {
  background: "rgba(12,18,28,.95)",
  border: "1px solid rgba(190,215,245,.12)",
  borderRadius: 15,
  padding: 16,
};

const buttonBase = {
  border: "1px solid rgba(190,215,245,.2)",
  background: "#182230",
  color: "#e8f1fa",
  borderRadius: 8,
  padding: "11px 14px",
  cursor: "pointer",
  fontSize: 11,
  letterSpacing: ".07em",
  fontWeight: 600,
};

function Metric({ label, value, unit = "" }) {
  return (
    <div
      style={{
        background: "#111a26",
        border: "1px solid #273343",
        borderRadius: 10,
        padding: 12,
      }}
    >
      <div
        style={{
          color: "#92a5b8",
          fontSize: 10,
          letterSpacing: 1,
          marginBottom: 8,
        }}
      >
        {label}
      </div>

      <div
        style={{
          fontSize: 21,
          fontWeight: 650,
          fontVariantNumeric: "tabular-nums",
        }}
      >
        {value}
        <span
          style={{
            fontSize: 11,
            color: "#94a7b8",
            marginLeft: 4,
          }}
        >
          {unit}
        </span>
      </div>
    </div>
  );
}

function Actuator({ name, value, max = 1, unit = "" }) {
  const numeric = Number(value) || 0;
  const percent = clamp(
    Math.abs(numeric) / max * 100,
    0,
    100
  );

  return (
    <div style={{ marginBottom: 13 }}>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          fontSize: 11,
          marginBottom: 7,
          gap: 12,
        }}
      >
        <span style={{ color: "#9cadbf" }}>{name}</span>
        <span>{n(numeric, 2)} {unit}</span>
      </div>

      <div
        style={{
          height: 5,
          borderRadius: 5,
          background: "#263546",
          overflow: "hidden",
        }}
      >
        <div
          style={{
            height: "100%",
            width: `${percent}%`,
            background:
              "linear-gradient(90deg,#4b82b7,#cde8ff)",
            transition: "width .15s linear",
          }}
        />
      </div>
    </div>
  );
}

function Attitude({ quaternion }) {
  const [w, x, y, z] =
    Array.isArray(quaternion) &&
    quaternion.length === 4
      ? quaternion.map(Number)
      : [1, 0, 0, 0];

  const toDeg = 180 / Math.PI;

  const roll = Math.atan2(
    2 * (w * x + y * z),
    1 - 2 * (x * x + y * y)
  ) * toDeg;

  const pitch = Math.asin(
    clamp(2 * (w * y - z * x), -1, 1)
  ) * toDeg;

  const yaw = Math.atan2(
    2 * (w * z + x * y),
    1 - 2 * (y * y + z * z)
  ) * toDeg;

  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "repeat(3,1fr)",
        gap: 8,
      }}
    >
      <Metric label="ROLL" value={n(roll)} unit="°" />
      <Metric label="PITCH" value={n(pitch)} unit="°" />
      <Metric label="YAW" value={n(yaw)} unit="°" />
    </div>
  );
}

export default function Exp005() {
  const [data, setData] = useState(INITIAL);
  const [connected, setConnected] = useState(false);
  const [paused, setPaused] = useState(false);
  const [autoRerun, setAutoRerun] = useState(true);
  const [speed, setSpeed] = useState(1);
  const [camera, setCamera] = useState("cinematic");
  const [message, setMessage] = useState("CONNECTING");
  const [result, setResult] = useState(null);
  const [episode, setEpisode] = useState(0);

  const socketRef = useRef(null);
  const initialFuelRef = useRef(null);
  const autoRerunRef = useRef(true);
  const speedRef = useRef(1);

  function send(command, extra = {}) {
    const socket = socketRef.current;

    if (socket?.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify({
        command,
        ...extra,
      }));
    }
  }

  useEffect(() => {
    let active = true;
    let socket = null;
    let reconnectTimer = null;

    const connect = () => {
      if (!active) return;

      socket = new WebSocket(WS);
      socketRef.current = socket;

      socket.onopen = () => {
        if (!active) return;

        setConnected(true);
        setMessage("LIVE TELEMETRY");

        socket.send(JSON.stringify({
          command: "auto_rerun",
          enabled: autoRerunRef.current,
        }));

        socket.send(JSON.stringify({
          command: "speed",
          value: speedRef.current,
        }));
      };

      socket.onmessage = (event) => {
        if (!active) return;

        let packet;

        try {
          packet = JSON.parse(event.data);
        } catch {
          return;
        }

        switch (packet.type) {
          case "telemetry":
            setData((previous) => ({
              ...previous,
              ...packet,
            }));
            break;

          case "episode_started":
            setEpisode(packet.episode);
            setResult(null);
            setPaused(false);
            setMessage("MISSION RUNNING");
            initialFuelRef.current = null;

            setData({
              ...INITIAL,
              ...packet,
              episode: packet.episode,
              phase: packet.phase || "INITIALIZING",
            });
            break;

          case "episode_complete":
            setResult(packet);
            setMessage(
              packet.success
                ? "LANDING SUCCESS"
                : "EPISODE COMPLETE"
            );
            break;

          case "control_state":
            setPaused(Boolean(packet.paused));
            setSpeed(packet.playback_speed ?? 1);
            setAutoRerun(Boolean(packet.auto_rerun));
            break;

          case "error":
            setMessage(
              packet.message || "BACKEND ERROR"
            );
            break;

          default:
            break;
        }
      };

      socket.onerror = () => {
        if (active) setMessage("CONNECTION ERROR");
      };

      socket.onclose = () => {
        if (!active) return;

        setConnected(false);
        setMessage("RECONNECTING");

        reconnectTimer = setTimeout(connect, 1500);
      };
    };

    connect();

    return () => {
      active = false;
      clearTimeout(reconnectTimer);
      socket?.close();
    };
  }, []);

  useEffect(() => {
    const fuel = Number(data.fuel_kg);

    if (
      initialFuelRef.current === null &&
      data.type === "telemetry" &&
      Number.isFinite(fuel) &&
      fuel > 0
    ) {
      initialFuelRef.current = fuel;
    }
  }, [data]);

  const fuelPct = initialFuelRef.current
    ? clamp(
        Number(data.fuel_kg) /
        initialFuelRef.current * 100,
        0,
        100
      )
    : 0;

  const flaps = data.flap_angles_deg || [0, 0, 0, 0];
  const brakes =
    data.airbrake_deployment || [0, 0, 0, 0];
  const gimbal =
    data.gimbal_angles_deg || [0, 0];

  const heading = {
    color: "#dce8f5",
    fontSize: 11,
    fontWeight: 700,
    letterSpacing: 1.4,
    marginBottom: 16,
  };

  return (
    <div
      className="zero004"
      style={{
        minHeight: "100vh",
        background: "#05090f",
        color: "#f1f6fc",
        padding: 20,
      }}
    >
      <header
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          flexWrap: "wrap",
          gap: 15,
          marginBottom: 18,
        }}
      >
        <div>
          <div
            style={{
              fontSize: 12,
              letterSpacing: 3,
              fontWeight: 700,
            }}
          >
            ◯ ZERO SPACE
          </div>

          <h1
            style={{
              margin: "8px 0 5px",
              fontSize: 28,
            }}
          >
            EXP005 / STAGE 2
          </h1>

          <div
            style={{
              color: "#8b9caf",
              fontSize: 11,
              letterSpacing: 1,
            }}
          >
            AUTONOMOUS 6DOF MISSION CONTROL
          </div>
        </div>

        <div
          style={{
            ...panel,
            display: "flex",
            gap: 10,
            alignItems: "center",
          }}
        >
          <span
            style={{
              height: 9,
              width: 9,
              borderRadius: "50%",
              background: connected
                ? "#39d5a0"
                : "#ff6876",
            }}
          />

          <span style={{ fontSize: 11 }}>
            {message}
          </span>
        </div>
      </header>

      {/* MISSION CONTROL */}
      <section style={{ ...panel, marginBottom: 16 }}>
        <div style={heading}>MISSION OPERATIONS</div>

        <div
          style={{
            display: "flex",
            flexWrap: "wrap",
            gap: 10,
            alignItems: "center",
          }}
        >
          <button
            style={{
              ...buttonBase,
              background: "#dce9f7",
              color: "#0b1520",
            }}
            disabled={!connected}
            onClick={() => {
              setResult(null);
              setMessage("RESTARTING MISSION");
              send("rerun");
            }}
          >
            ↻ RERUN MISSION
          </button>

          <button
            style={buttonBase}
            disabled={!connected || Boolean(result)}
            onClick={() => send(
              paused ? "resume" : "pause"
            )}
          >
            {paused ? "▶ RESUME" : "Ⅱ PAUSE"}
          </button>

          <label
            style={{
              display: "flex",
              gap: 8,
              alignItems: "center",
              fontSize: 11,
              color: "#b5c4d4",
            }}
          >
            SPEED

            <select
              value={speed}
              disabled={!connected}
              onChange={(e) => {
                const next = Number(e.target.value);
                speedRef.current = next;
                setSpeed(next);
                send("speed", { value: next });
              }}
              style={{
                ...buttonBase,
                padding: "9px 10px",
              }}
            >
              {[0.5, 1, 2, 4].map((value) => (
                <option key={value} value={value}>
                  {value}×
                </option>
              ))}
            </select>
          </label>

          <label
            style={{
              display: "flex",
              gap: 8,
              alignItems: "center",
              fontSize: 11,
              color: "#b5c4d4",
              cursor: "pointer",
            }}
          >
            <input
              type="checkbox"
              checked={autoRerun}
              disabled={!connected}
              onChange={(e) => {
                const enabled = e.target.checked;
                autoRerunRef.current = enabled;
                setAutoRerun(enabled);
                send("auto_rerun", { enabled });
              }}
            />
            AUTO-RERUN
          </label>

          <div
            style={{
              marginLeft: "auto",
              fontSize: 11,
              color: "#a7b9ca",
            }}
          >
            EPISODE {episode || data.episode || 0}
            {"  •  "}
            T+ {n(data.time_s)} s
          </div>
        </div>
      </section>

      <div
        style={{
          display: "grid",
          gridTemplateColumns:
            "minmax(0, 1fr) minmax(290px, 340px)",
          gap: 16,
          alignItems: "start",
        }}
      >
        <section
          style={{
            ...panel,
            padding: 0,
            overflow: "hidden",
          }}
        >
          <div
            style={{
              padding: "13px 16px",
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              gap: 10,
              flexWrap: "wrap",
              borderBottom: "1px solid #24303d",
            }}
          >
            <span
              style={{
                fontSize: 11,
                letterSpacing: 1,
              }}
            >
              LIVE FLIGHT VISUALIZATION
            </span>

            <div
              style={{
                display: "flex",
                flexWrap: "wrap",
                gap: 6,
              }}
            >
              {[
                ["cinematic", "CINEMATIC"],
                ["launchpad", "LAUNCHPAD"],
                ["follow", "FOLLOW"],
                ["earth", "EARTH"],
              ].map(([id, label]) => (
                <button
                  key={id}
                  onClick={() => setCamera(id)}
                  style={{
                    ...buttonBase,
                    padding: "8px 10px",
                    background:
                      camera === id
                        ? "#e4eef9"
                        : "#15202d",
                    color:
                      camera === id
                        ? "#07111b"
                        : "#c5d4e4",
                  }}
                >
                  {label}
                </button>
              ))}
            </div>
          </div>

          <div
            className="zero-scene"
            style={{
              height: "min(70vh,700px)",
              minHeight: 440,
              position: "relative",
            }}
          >
            <RocketScene
              telemetry={data}
              cameraMode={camera}
              episode={episode}
            />

            <div
              style={{
                position: "absolute",
                left: 16,
                bottom: 16,
                padding: "11px 14px",
                background: "rgba(0,0,0,.65)",
                borderRadius: 9,
                fontSize: 12,
                pointerEvents: "none",
              }}
            >
              <div
                style={{
                  color: "#91a7bd",
                  fontSize: 10,
                  marginBottom: 5,
                }}
              >
                FLIGHT PHASE
              </div>
              {data.phase}
            </div>
          </div>

          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              flexWrap: "wrap",
              gap: 10,
              padding: 14,
              color: "#a8b9ca",
              fontSize: 11,
            }}
          >
            <span>PPO 6DOF AUTOPILOT</span>
            <span>ALT {n(data.altitude_m, 0)} m</span>
            <span>VZ {n(data.vertical_velocity_mps)} m/s</span>
          </div>
        </section>

        <aside
          style={{
            display: "flex",
            flexDirection: "column",
            gap: 14,
          }}
        >
          <div style={panel}>
            <div style={heading}>FLIGHT TELEMETRY</div>
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "1fr 1fr",
                gap: 9,
              }}
            >
              <Metric
                label="ALTITUDE"
                value={n(data.altitude_m, 0)}
                unit="m"
              />
              <Metric
                label="SPEED"
                value={n(data.speed_mps)}
                unit="m/s"
              />
              <Metric
                label="VERTICAL SPEED"
                value={n(data.vertical_velocity_mps)}
                unit="m/s"
              />
              <Metric
                label="ACCELERATION"
                value={n(data.acceleration_mps2)}
                unit="m/s²"
              />
              <Metric
                label="DYNAMIC PRESSURE"
                value={n(data.dynamic_pressure_pa, 0)}
                unit="Pa"
              />
              <Metric
                label="MASS"
                value={n(data.mass_kg, 0)}
                unit="kg"
              />
            </div>
          </div>

          <div style={panel}>
            <div style={heading}>VEHICLE ATTITUDE</div>
            <Attitude
              quaternion={data.orientation_quat}
            />
          </div>

          <div style={panel}>
            <div style={heading}>PROPULSION</div>
            <Actuator
              name="ENGINE THROTTLE"
              value={Number(data.throttle) * 100}
              max={100}
              unit="%"
            />
            <Actuator
              name="FUEL REMAINING"
              value={fuelPct}
              max={100}
              unit="%"
            />
            <div
              style={{
                color: "#8ca0b5",
                fontSize: 11,
              }}
            >
              {n(data.fuel_kg, 0)} kg propellant
            </div>
          </div>
        </aside>
      </div>

      <div
        style={{
          display: "grid",
          gridTemplateColumns:
            "repeat(auto-fit,minmax(250px,1fr))",
          gap: 16,
          marginTop: 16,
        }}
      >
        <section style={panel}>
          <div style={heading}>FLAP CONTROL</div>
          {flaps.map((v, i) => (
            <Actuator
              key={i}
              name={`FLAP ${i + 1}`}
              value={v}
              max={30}
              unit="°"
            />
          ))}
        </section>

        <section style={panel}>
          <div style={heading}>AIRBRAKE CONTROL</div>
          {brakes.map((v, i) => (
            <Actuator
              key={i}
              name={`AIRBRAKE ${i + 1}`}
              value={Number(v) * 100}
              max={100}
              unit="%"
            />
          ))}
        </section>

        <section style={panel}>
          <div style={heading}>THRUST VECTOR / RCS</div>
          <Actuator
            name="GIMBAL X"
            value={gimbal[0]}
            max={5}
            unit="°"
          />
          <Actuator
            name="GIMBAL Y"
            value={gimbal[1]}
            max={5}
            unit="°"
          />
          <Actuator
            name="RCS"
            value={Number(data.rcs_command) * 100}
            max={100}
            unit="%"
          />
        </section>
      </div>

      {result && (
        <section style={{ ...panel, marginTop: 16 }}>
          <div style={heading}>EPISODE RESULT</div>
          <div style={{ fontSize: 13 }}>
            Episode {result.episode}
            {" • "}
            {result.success
              ? "SUCCESSFUL LANDING"
              : "LANDING NOT CONFIRMED"}
            {" • "}
            {n(result.time_s)} s
            {" • "}
            Reward {n(result.total_reward)}
          </div>

          <button
            style={{
              ...buttonBase,
              marginTop: 14,
            }}
            onClick={() => {
              setResult(null);
              send("rerun");
            }}
          >
            ↻ RUN AGAIN
          </button>
        </section>
      )}

      <footer
        style={{
          padding: "20px 0",
          color: "#6e8196",
          fontSize: 10,
          letterSpacing: 1,
        }}
      >
        ZERO SPACE / EXP005 / STAGE 2
        {" • "}
        AUTONOMOUS FLIGHT RESEARCH
        {" • "}
        SIMULATION
      </footer>
    </div>
  );
}
