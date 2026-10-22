
import { useEffect, useRef, useState } from "react";
import {
  Activity,
  Gauge,
  RotateCcw,
  Play,
  Pause,
  Radio,
  Cpu,
  Crosshair,
  Flame,
  Clock3,
  Video,
} from "lucide-react";

import RocketScene from "./RocketScene.jsx";
import FlightRecorder from "./FlightRecorder.jsx";
import "./styles.css";

const API = "http://127.0.0.1:8000";
const WS = "ws://127.0.0.1:8000";

const ENGINE_NAMES = [
  "COASTING",
  "LEFT THRUSTER",
  "MAIN ENGINE",
  "RIGHT THRUSTER",
];

function Metric({ label, value, unit = "" }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>
        {value} <small>{unit}</small>
      </strong>
    </div>
  );
}

function Panel({ title, icon: Icon, children }) {
  return (
    <section className="panel">
      <div className="panel-title">
        <Icon size={15} strokeWidth={1.6} />
        <span>{title}</span>
      </div>
      {children}
    </section>
  );
}

function format(value, digits = 2) {
  return typeof value === "number"
    ? value.toFixed(digits)
    : "—";
}

export default function App() {
  const [flight, setFlight] = useState(null);
  const [models, setModels] = useState(["baseline"]);
  const [selectedModel, setSelectedModel] = useState("baseline");
  const [connected, setConnected] = useState(false);
  const [paused, setPaused] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [history, setHistory] = useState([]);
  const [error, setError] = useState("");
  const [cinematic, setCinematic] = useState(false);

  const socket = useRef(null);

  useEffect(() => {
    fetch(`${API}/api/status`)
      .then((response) => response.json())
      .then((data) => {
        if (data.available_models?.length) {
          setModels(data.available_models);

          if (!data.available_models.includes(selectedModel)) {
            setSelectedModel(data.available_models[0]);
          }
        }
      })
      .catch(() => setError("Backend is not responding"));
  }, []);

  useEffect(() => {
    setConnected(false);
    setFlight(null);
    setPaused(false);
    setHistory([]);
    setError("");

    const connection = new WebSocket(
      `${WS}/ws/flight/${selectedModel}`
    );

    socket.current = connection;

    connection.onopen = () => {
      setConnected(true);
      connection.send(
        JSON.stringify({
          command: "speed",
          value: speed,
        })
      );
    };

    connection.onmessage = (event) => {
      const message = JSON.parse(event.data);

      if (message.type === "telemetry") {
        setFlight(message);
      }

      if (message.type === "episode_complete") {
        setHistory((previous) => [
          message,
          ...previous,
        ].slice(0, 10));
      }

      if (message.type === "error") {
        setError(message.message);
      }
    };

    connection.onclose = () => setConnected(false);

    connection.onerror = () =>
      setError("WebSocket connection failed");

    return () => {
      connection.close();
      socket.current = null;
    };
  }, [selectedModel]);

  function sendCommand(command, value) {
    if (socket.current?.readyState === WebSocket.OPEN) {
      socket.current.send(JSON.stringify({ command, value }));
    }
  }

  function togglePlayback() {
    sendCommand(paused ? "play" : "pause");
    setPaused(!paused);
  }

  function changeSpeed(value) {
    setSpeed(value);
    sendCommand("speed", value);
  }

  const averageReward = history.length
    ? history.reduce(
        (sum, item) => sum + item.reward,
        0
      ) / history.length
    : null;

  return (
    <div className="mission-control">
      <header className="topbar">
        <div className="brand">
          <div className="brand-symbol">Z</div>
          <div>
            <h1>ZERO SPACE</h1>
            <p>EXPERIMENT 001 / AUTONOMOUS LANDING</p>
          </div>
        </div>

        <div className="header-center">
          <span className="eyebrow">
            FLIGHT CONTROL SYSTEM
          </span>
          <strong>MISSION CONTROL</strong>
        </div>

        <div className={`connection ${connected ? "online" : ""}`}>
          <span className="status-dot" />
          {connected ? "TELEMETRY LIVE" : "DISCONNECTED"}
        </div>
      </header>

      {error && <div className="error-banner">{error}</div>}

      <main className="dashboard-grid">
        <aside className="sidebar">
          <Panel title="FLIGHT TELEMETRY" icon={Activity}>
            <Metric
              label="Horizontal position"
              value={format(flight?.position?.x)}
              unit="norm."
            />
            <Metric
              label="Vertical position"
              value={format(flight?.position?.y)}
              unit="norm."
            />
            <Metric
              label="Vertical velocity"
              value={format(flight?.velocity?.y)}
              unit="norm."
            />
            <Metric
              label="Horizontal velocity"
              value={format(flight?.velocity?.x)}
              unit="norm."
            />
            <Metric
              label="Tilt"
              value={format(
                (flight?.angle ?? 0) * 180 / Math.PI
              )}
              unit="deg"
            />
            <Metric
              label="Angular velocity"
              value={format(flight?.angular_velocity)}
              unit="rad/s*"
            />
          </Panel>

          <Panel title="ENGINE CONTROL" icon={Flame}>
            <div className="engine-state">
              <span className="engine-light" />
              <div>
                <small>ACTIVE COMMAND</small>
                <strong>
                  {ENGINE_NAMES[flight?.action ?? 0]}
                </strong>
              </div>
            </div>
            <p className="muted">
              Engine selection is generated by the trained PPO policy.
            </p>
          </Panel>

          <Panel title="LANDING CONTACT" icon={Crosshair}>
            <Metric
              label="Left landing leg"
              value={flight?.legs?.left ? "CONTACT" : "AIRBORNE"}
            />
            <Metric
              label="Right landing leg"
              value={flight?.legs?.right ? "CONTACT" : "AIRBORNE"}
            />
          </Panel>
        </aside>

        <section className="viewport">
          <div className="viewport-top">
            <span>CAM 01 / EXTERNAL TRACKING</span>
            <span>GYMNASIUM SIMULATION</span>
          </div>

          <RocketScene
            flight={flight}
            cinematic={cinematic}
          />

          <div className="viewport-overlay">
            <span className="eyebrow">ACTIVE FLIGHT</span>
            <strong>
              EPISODE {String(flight?.episode ?? 0).padStart(3, "0")}
            </strong>
            <span>STEP {flight?.step ?? 0}</span>
          </div>

          <div className="viewport-bottom">
            <span>
              <Radio size={13} />
              PPO AUTOPILOT
            </span>
            <span>
              {flight?.finished
                ? "FLIGHT COMPLETE"
                : "SIMULATION ACTIVE"}
            </span>
          </div>
        </section>

        <aside className="sidebar">
          <Panel title="SIMULATION CONTROL" icon={Cpu}>
            <label className="field-label">TRAINED POLICY</label>
            <select
              value={selectedModel}
              onChange={(event) =>
                setSelectedModel(event.target.value)
              }
            >
              {models.map((name) => (
                <option key={name} value={name}>
                  {name.toUpperCase().replaceAll("_", " ")}
                </option>
              ))}
            </select>

            <div className="controls">
              <button
                onClick={togglePlayback}
                disabled={!connected}
              >
                {paused ? <Play size={16} /> : <Pause size={16} />}
                {paused ? "RESUME" : "PAUSE"}
              </button>

              <button
                onClick={() => {
                  sendCommand("restart");
                  setPaused(false);
                }}
                disabled={!connected}
              >
                <RotateCcw size={16} />
                RESTART
              </button>
            </div>

            <label className="field-label">PLAYBACK SPEED</label>
            <div className="speed-options">
              {[0.5, 1, 2, 4].map((value) => (
                <button
                  key={value}
                  className={speed === value ? "selected" : ""}
                  onClick={() => changeSpeed(value)}
                >
                  {value}×
                </button>
              ))}
            </div>

            <button
              onClick={() => setCinematic((value) => !value)}
              className="cinematic-toggle"
            >
              {cinematic
                ? "DISABLE CINEMATIC CAMERA"
                : "ENABLE CINEMATIC CAMERA"}
            </button>
          </Panel>

          <Panel title="MISSION PERFORMANCE" icon={Gauge}>
            <Metric
              label="Current reward"
              value={format(flight?.total_reward, 1)}
            />
            <Metric
              label="Recent avg. reward"
              value={format(averageReward, 1)}
            />
            <Metric
              label="Flight duration"
              value={format(flight?.sim_time, 1)}
              unit="s"
            />
            <Metric
              label="Simulation steps"
              value={flight?.step ?? 0}
            />
          </Panel>

          <Panel title="RECENT FLIGHTS" icon={Clock3}>
            {history.length === 0 ? (
              <p className="muted">
                Waiting for first completed flight...
              </p>
            ) : (
              history.slice(0, 5).map((item) => (
                <div className="history-row" key={item.episode}>
                  <span>
                    FLIGHT {String(item.episode).padStart(3, "0")}
                  </span>
                  <strong>{format(item.reward, 1)}</strong>
                </div>
              ))
            )}
          </Panel>

          <Panel title="VIDEO CAPTURE" icon={Video}>
            <FlightRecorder />
          </Panel>
        </aside>
      </main>

      <footer className="footer">
        <span>ZERO SPACE / EXPERIMENTAL RESEARCH SYSTEM</span>
        <span>SIMULATED PHYSICS · NOT A REAL FLIGHT CONTROLLER</span>
        <span>EXP 001 / BUILD 0.2</span>
      </footer>
    </div>
  );
}
