
import { useEffect, useRef, useState } from "react";
import RocketScene from "./exp004/RocketScene.jsx";
import "./exp004.css";

const DEFAULT_SERVER =
  "ws://127.0.0.1:8004/ws/flight";

const INITIAL_DATA = {
  altitude_m: 0,
  max_altitude_m: 0,
  speed_mps: 0,
  vertical_velocity_mps: 0,
  acceleration_mps2: 0,
  fuel_kg: 180000,
  mass_kg: 200000,
  throttle: 0,
  dynamic_pressure_pa: 0,
  heating_proxy: 0,
  time_s: 0,
  total_reward: 0,
  phase: "ON PAD",
  controller: "PPO AUTOPILOT",
  episode: 1,
};

function formatNumber(value, digits = 1) {
  const number = Number(value);

  if (!Number.isFinite(number)) {
    return "0";
  }

  return number.toLocaleString("en-US", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

function Metric({ label, value, unit = "" }) {
  return (
    <div
      style={{
        padding: "14px 16px",
        border: "1px solid rgba(255,255,255,0.12)",
        borderRadius: 12,
        background: "rgba(255,255,255,0.035)",
      }}
    >
      <div
        style={{
          fontSize: 11,
          letterSpacing: 1.2,
          color: "#9ca3af",
          marginBottom: 8,
        }}
      >
        {label}
      </div>

      <div
        style={{
          fontSize: 23,
          fontWeight: 600,
          color: "#f5f5f5",
          fontVariantNumeric: "tabular-nums",
        }}
      >
        {value}
        {unit && (
          <span
            style={{
              fontSize: 12,
              color: "#9ca3af",
              marginLeft: 6,
            }}
          >
            {unit}
          </span>
        )}
      </div>
    </div>
  );
}

export default function Exp004({
  serverUrl = DEFAULT_SERVER,
  experiment = "004",
  title = "AUTONOMOUS LIFTOFF",
  targetAltitude = 10000,
}) {
  const [data, setData] = useState(INITIAL_DATA);
  const [connected, setConnected] = useState(false);
  const [message, setMessage] = useState(
    "Connecting to Mission Control..."
  );
  const [cameraMode, setCameraMode] = useState("launchpad");
  const [lastResult, setLastResult] = useState(null);

  const socketRef = useRef(null);

  useEffect(() => {
    let active = true;
    let reconnectTimer = null;
    let socket = null;

    setConnected(false);
    setData(INITIAL_DATA);
    setLastResult(null);
    setMessage("Connecting to Mission Control...");

    function connect() {
      if (!active) return;

      socket = new WebSocket(serverUrl);
      socketRef.current = socket;

      socket.onopen = () => {
        if (!active) return;

        setConnected(true);
        setMessage("AUTOPILOT CONNECTED");
      };

      socket.onmessage = (event) => {
        if (!active) return;

        let packet;

        try {
          packet = JSON.parse(event.data);
        } catch {
          return;
        }

        if (packet.type === "telemetry") {
          setData((previous) => ({
            ...previous,
            ...packet,
          }));

          setMessage(
            packet.controller ||
              packet.phase ||
              "AUTOPILOT ACTIVE"
          );
        }

        if (packet.type === "status") {
          setMessage(
            packet.message || "AUTOPILOT ACTIVE"
          );
        }

        if (packet.type === "episode_complete") {
          setLastResult(packet);

          setMessage(
            packet.success
              ? "MISSION SUCCESS"
              : "MISSION COMPLETE — TARGET NOT ACHIEVED"
          );
        }

        if (packet.type === "error") {
          setMessage(
            packet.message || "SERVER ERROR"
          );
        }
      };

      socket.onerror = () => {
        if (active) {
          setMessage("CONNECTION ERROR");
        }
      };

      socket.onclose = () => {
        if (!active) return;

        setConnected(false);
        setMessage("RECONNECTING TO MISSION CONTROL");

        reconnectTimer = setTimeout(connect, 2000);
      };
    }

    connect();

    return () => {
      active = false;

      if (reconnectTimer) {
        clearTimeout(reconnectTimer);
      }

      if (socket) {
        socket.close();
      }

      socketRef.current = null;
    };
  }, [serverUrl]);

  const altitude = Number(data.altitude_m) || 0;
  const throttle = Number(data.throttle) || 0;
  const fuel = Number(data.fuel_kg) || 0;

  const progress = Math.min(
    100,
    Math.max(0, (altitude / targetAltitude) * 100)
  );

  const noLiftoff =
    connected &&
    Number(data.time_s) > 10 &&
    Number(data.max_altitude_m) < 1;

  const isExp005 = experiment === "005";

  return (
    <div
      className="zero004"
      style={{
        minHeight: "100vh",
        background: "#080b10",
        color: "#f4f4f5",
        padding: 22,
        fontFamily:
          "Inter, system-ui, -apple-system, sans-serif",
      }}
    >
      <header
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          flexWrap: "wrap",
          gap: 16,
          marginBottom: 20,
        }}
      >
        <div>
          <div
            style={{
              fontSize: 12,
              letterSpacing: 3,
              color: "#a1a1aa",
              marginBottom: 8,
            }}
          >
            ZERO SPACE / EXPERIMENT {experiment}
          </div>

          <h1
            style={{
              margin: 0,
              fontSize: 28,
              letterSpacing: 1,
              fontWeight: 650,
            }}
          >
            {title}
          </h1>

          <div
            style={{
              color: "#9ca3af",
              fontSize: 12,
              marginTop: 8,
            }}
          >
            PHYSICS SIMULATION · PPO AUTOPILOT · LIVE TELEMETRY
          </div>
        </div>

        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 10,
            padding: "10px 14px",
            borderRadius: 10,
            border: "1px solid rgba(255,255,255,0.12)",
          }}
        >
          <span
            style={{
              width: 9,
              height: 9,
              borderRadius: "50%",
              background: connected ? "#22c55e" : "#ef4444",
              boxShadow: connected
                ? "0 0 12px #22c55e"
                : "none",
            }}
          />

          <span style={{ fontSize: 12 }}>
            {connected ? "CONNECTED" : "DISCONNECTED"}
          </span>
        </div>
      </header>

      <div
        style={{
          display: "grid",
          gridTemplateColumns:
            "minmax(0, 1.65fr) minmax(300px, 1fr)",
          gap: 18,
          alignItems: "start",
        }}
      >
        <section
          style={{
            minWidth: 0,
            border: "1px solid rgba(255,255,255,0.12)",
            borderRadius: 16,
            overflow: "hidden",
            background: "#101722",
          }}
        >
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              flexWrap: "wrap",
              padding: "12px 16px",
              borderBottom:
                "1px solid rgba(255,255,255,0.10)",
              gap: 10,
            }}
          >
            <span
              style={{
                fontSize: 12,
                letterSpacing: 1.5,
              }}
            >
              3D FLIGHT VISUALIZATION
            </span>

            <div style={{ display: "flex", gap: 6 }}>
              {["launchpad", "follow", "earth"].map(
                (mode) => (
                  <button
                    key={mode}
                    onClick={() => setCameraMode(mode)}
                    style={{
                      padding: "7px 10px",
                      borderRadius: 7,
                      border:
                        "1px solid rgba(255,255,255,0.15)",
                      background:
                        cameraMode === mode
                          ? "#e5e7eb"
                          : "transparent",
                      color:
                        cameraMode === mode
                          ? "#111827"
                          : "#d1d5db",
                      cursor: "pointer",
                      fontSize: 11,
                      textTransform: "uppercase",
                    }}
                  >
                    {mode}
                  </button>
                )
              )}
            </div>
          </div>

          <div style={{ height: 570, position: "relative" }}>
            <RocketScene
              telemetry={data}
              data={data}
              cameraMode={cameraMode}
              mode={cameraMode}
            />

            <div
              style={{
                position: "absolute",
                left: 16,
                bottom: 16,
                background: "rgba(0,0,0,0.55)",
                borderRadius: 8,
                padding: "10px 14px",
                pointerEvents: "none",
              }}
            >
              <div
                style={{
                  fontSize: 10,
                  color: "#a1a1aa",
                  letterSpacing: 1,
                }}
              >
                FLIGHT PHASE
              </div>

              <div
                style={{
                  fontSize: 16,
                  fontWeight: 600,
                  marginTop: 5,
                }}
              >
                {data.phase}
              </div>
            </div>
          </div>

          <div
            style={{
              padding: 16,
              borderTop:
                "1px solid rgba(255,255,255,0.10)",
            }}
          >
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                marginBottom: 9,
                fontSize: 12,
              }}
            >
              <span>ALTITUDE TARGET</span>
              <span>
                {formatNumber(altitude, 0)} /{" "}
                {formatNumber(targetAltitude, 0)} m
              </span>
            </div>

            <div
              style={{
                height: 7,
                background: "#252b35",
                borderRadius: 20,
                overflow: "hidden",
              }}
            >
              <div
                style={{
                  height: "100%",
                  width: `${progress}%`,
                  background: "#e5e7eb",
                  transition: "width 0.3s",
                }}
              />
            </div>
          </div>
        </section>

        <aside style={{ minWidth: 0 }}>
          <div
            style={{
              border: "1px solid rgba(255,255,255,0.12)",
              borderRadius: 14,
              padding: 16,
              marginBottom: 14,
              background: "#10141b",
            }}
          >
            <div
              style={{
                fontSize: 11,
                letterSpacing: 1.5,
                color: "#9ca3af",
                marginBottom: 8,
              }}
            >
              MISSION STATUS
            </div>

            <div style={{ fontSize: 14, fontWeight: 600 }}>
              {message}
            </div>

            <div
              style={{
                marginTop: 10,
                fontSize: 12,
                color: "#a1a1aa",
              }}
            >
              Episode {data.episode || 1} ·{" "}
              {data.controller || "PPO AUTOPILOT"}
            </div>
          </div>

          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(2, minmax(0, 1fr))",
              gap: 10,
            }}
          >
            <Metric
              label="ALTITUDE"
              value={formatNumber(altitude)}
              unit="m"
            />

            <Metric
              label="MAX ALTITUDE"
              value={formatNumber(data.max_altitude_m)}
              unit="m"
            />

            <Metric
              label="SPEED"
              value={formatNumber(data.speed_mps)}
              unit="m/s"
            />

            <Metric
              label="VERTICAL VELOCITY"
              value={formatNumber(
                data.vertical_velocity_mps ?? 0
              )}
              unit="m/s"
            />

            <Metric
              label="THROTTLE"
              value={formatNumber(throttle * 100, 0)}
              unit="%"
            />

            <Metric
              label="FUEL"
              value={formatNumber(fuel, 0)}
              unit="kg"
            />

            <Metric
              label="ACCELERATION"
              value={formatNumber(
                data.acceleration_mps2
              )}
              unit="m/s²"
            />

            <Metric
              label="DYNAMIC PRESSURE"
              value={formatNumber(
                data.dynamic_pressure_pa,
                0
              )}
              unit="Pa"
            />

            <Metric
              label="FLIGHT TIME"
              value={formatNumber(data.time_s)}
              unit="s"
            />

            <Metric
              label="TOTAL REWARD"
              value={formatNumber(data.total_reward)}
            />
          </div>

          {noLiftoff && (
            <div
              style={{
                marginTop: 14,
                padding: 14,
                border: "1px solid #854d0e",
                borderRadius: 10,
                background: "rgba(133,77,14,0.12)",
                color: "#fbbf24",
                fontSize: 12,
              }}
            >
              NO LIFTOFF DETECTED — CHECK PPO THROTTLE
            </div>
          )}

          {lastResult && (
            <div
              style={{
                marginTop: 14,
                padding: 16,
                border:
                  "1px solid rgba(255,255,255,0.15)",
                borderRadius: 12,
              }}
            >
              <strong>
                {lastResult.success
                  ? "MISSION SUCCESS"
                  : "MISSION ENDED"}
              </strong>

              <p
                style={{
                  marginBottom: 0,
                  color: "#a1a1aa",
                  fontSize: 12,
                }}
              >
                Maximum altitude:{" "}
                {formatNumber(
                  lastResult.max_altitude_m,
                  0
                )}{" "}
                m
                {isExp005 &&
                  ` · Space reached: ${
                    lastResult.reached_space ? "YES" : "NO"
                  } · Landed: ${
                    lastResult.landed ? "YES" : "NO"
                  }`}
              </p>
            </div>
          )}
        </aside>
      </div>
    </div>
  );
}
