# ZERO SPACE

**An experimental rocket simulation and reinforcement learning research platform.**

ZERO SPACE is a Python-based aerospace simulation project with a React/Vite mission-control dashboard. It explores six-degree-of-freedom rigid-body dynamics, flight telemetry, attitude stabilization, and reinforcement learning for simulated rocket flight.

## Features

- Six-degree-of-freedom flight simulation
- Quaternion-based attitude representation
- Atmospheric drag, gravity, thrust, and fuel modeling
- PPO reinforcement learning experiments
- Flight telemetry and mission-control visualization
- Simulation diagnostics and evaluation tools

## Technology Stack

- Python
- NumPy
- Gymnasium
- Stable-Baselines3
- PyTorch
- FastAPI
- React and Vite

## Project Structure

```text
ZERO-SPACE/
├── configs/
├── dashboard/
├── experiments/
├── learning/
├── logs/
├── mission_control/
├── models/
├── physics/
├── results/
├── rocket/
├── requirements.txt
├── requirements-lock.txt
├── .gitignore
└── README.md
```

## Local Development

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/ZERO-SPACE.git
cd ZERO-SPACE
```

### 2. Create and activate a Python environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Start the simulation server

```powershell
python -m mission_control.exp005_server
```

### 5. Start the dashboard

Open another PowerShell terminal:

```powershell
cd dashboard
npm install
npm run dev
```

Open the local URL printed by Vite in your terminal.

> The exact server entry point and dashboard scripts must be verified against the current source before treating these commands as validated setup instructions.

## Research and Limitations

ZERO SPACE is an experimental educational simulation. Its simplified physics and reinforcement learning controllers are not validated for real-world rocket design, flight, or guidance.

Performance claims should be supported by reproducible simulation tests and documented evaluation results.

## Contributing

Contributions, bug reports, reproducible experiments, and improvements to simulation testing are welcome.

## License

License to be selected before public release.
