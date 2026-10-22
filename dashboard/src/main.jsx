
import React from "react";
import ReactDOM from "react-dom/client";

import Exp004 from "./Exp004.jsx";
import Exp005 from "./Exp005.jsx";

function Home() {
  const experiments = [
    {
      id: "004",
      title: "AUTONOMOUS LIFTOFF",
      description:
        "PPO-controlled rocket ascent to 10 km.",
      path: "/exp004",
    },
    {
      id: "005",
      title: "SPACEFLIGHT & RECOVERY",
      description:
        "Suborbital ascent beyond 100 km and powered landing attempt.",
      path: "/exp005",
    },
  ];

  return (
    <div
      style={{
        minHeight: "100vh",
        background: "#080b10",
        color: "#f5f5f5",
        padding: "60px 8%",
        fontFamily:
          "Inter, system-ui, -apple-system, sans-serif",
      }}
    >
      <div
        style={{
          color: "#a1a1aa",
          fontSize: 12,
          letterSpacing: 3,
          marginBottom: 12,
        }}
      >
        ZERO SPACE
      </div>

      <h1
        style={{
          fontSize: 44,
          fontWeight: 600,
          marginBottom: 10,
        }}
      >
        Mission Control
      </h1>

      <p
        style={{
          color: "#9ca3af",
          maxWidth: 650,
          lineHeight: 1.7,
          marginBottom: 35,
        }}
      >
        Autonomous rocket research experiments powered by
        physics simulation and reinforcement learning.
      </p>

      <div
        style={{
          display: "grid",
          gridTemplateColumns:
            "repeat(auto-fit, minmax(260px, 1fr))",
          gap: 18,
          maxWidth: 900,
        }}
      >
        {experiments.map((experiment) => (
          <a
            key={experiment.id}
            href={experiment.path}
            style={{
              textDecoration: "none",
              color: "inherit",
              border:
                "1px solid rgba(255,255,255,0.15)",
              background: "#11151d",
              borderRadius: 16,
              padding: 26,
              display: "block",
            }}
          >
            <div
              style={{
                color: "#9ca3af",
                letterSpacing: 2,
                fontSize: 12,
                marginBottom: 20,
              }}
            >
              EXPERIMENT {experiment.id}
            </div>

            <h2
              style={{
                fontSize: 20,
                fontWeight: 600,
                marginBottom: 12,
              }}
            >
              {experiment.title}
            </h2>

            <p
              style={{
                color: "#9ca3af",
                fontSize: 13,
                lineHeight: 1.7,
              }}
            >
              {experiment.description}
            </p>

            <div
              style={{
                marginTop: 24,
                fontSize: 12,
                letterSpacing: 1,
              }}
            >
              OPEN EXPERIMENT →
            </div>
          </a>
        ))}
      </div>
    </div>
  );
}

function App() {
  const path = window.location.pathname.replace(/\/$/, "");

  switch (path) {
    case "/exp004":
      return (
        <Exp004
          serverUrl="ws://127.0.0.1:8004/ws/flight"
          experiment="004"
          title="AUTONOMOUS LIFTOFF"
          targetAltitude={10000}
        />
      );

    case "/exp005":
      return <Exp005 />;

    default:
      return <Home />;
  }
}

ReactDOM.createRoot(
  document.getElementById("root")
).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
