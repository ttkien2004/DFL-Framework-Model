# BCFL CoCo Cluster Simulation Framework

This repository contains the simulation framework for the Decentralized
Federated Learning (DFL) system, integrating CoCo (Communication-Efficient
optimization), Blockchain (Committee Consensus Mechanism), Clustering (DFCA for
Non-IID data), and Privacy Preserving mechanisms (LDP + SMC).

## Project Structure

```bash
bcfl_project/
│
├── app/                        # Core application source code
│   ├── __init__.py             # Package initializer
│   ├── models/                 # Neural Network architectures
│   │   └── cnn.py              # CNN/VGG9 model definitions
│   ├── core/                   # DFL Entities logic
│   │   ├── worker.py           # Worker logic (Training, LDP, Clustering)
│   │   └── cluster_head.py     # Cluster Head logic (Aggregation, CoCo, BALANCE)
│   ├── blockchain/             # Blockchain module
│   │   ├── block.py            # Block structure definition
│   │   └── consensus.py        # Consensus logic (Voting, Committee)
│   └── utils/                  # Utility functions (Helpers, Cryptography)
│
├── templates/                  # HTML templates for the Web Dashboard
│   └── dashboard.html          # Real-time visualization UI
│
├── main.py                     # Entry point (Flask API Server)
├── client_summary.py           # Client script to fetch and display summary tables
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Docker image configuration
└── docker-compose.yml          # Docker services configuration

```

## Installation & Running

**Prerequisites**:

- `Docker` and `Docker Compose` installed on your machine.

**Option 1: Running with Docker**

This method ensures all dependencies and environments are set up:

1. **Build the image**:

```bash
docker-compose build
```

2. **Start the container**:

```bash
docker-composse up
```

To run in the background, add `-d`: `docker-compose up -d`

3. **Acce the application**:

- **API**: [http://localhost:5000](http://localhost:5000)

- **Dashboard**:
  [http://localhost:5000/dashboard](http://localhost:5000/dashboard)

**Option 2: Running locally**

If you prefer running without Docker:

1. **Create a virtual environmentt (optional) and install dependencies**:

```bash
pip install -r requirementts.txt
```

2. **Start the Flask server**:

```bash
python main.py
```

## API Endpoint

The simulation is controlled via a RESTful API. Below are the available
endpoints:

| Method | Endpoint     | Description                                                                                                  |
| ------ | ------------ | ------------------------------------------------------------------------------------------------------------ |
| GET    | /            | Health Check. Returns a welcome message to verify the API is running.                                        |
| GET    | /dashboard   | "Web UI. Renders the real-time HTML dashboard to monitor training progress (Accuracy, Loss, Block Height)."  |
| POST   | /run_round   | Trigger Training. Executes one full DFL round (Clustering → Training → Aggregation → Blockchain Consensus).  |
| GET    | /get_metrics | "Fetch Data. Returns the complete history of training metrics (rounds,accuracy, loss, etc.) in JSON format." |

## Usage Examples

1. **Triggering the Simulation**:

You need to trigger the training rounds manually. The system does not train
automatically unless requested.

**USing cURL (Optional)**:

```bash
# Run one round
curl -X POST http://localhost:5000/run_round
```

**Using PowerShell (Loop 20 rounds)**:

```bash
1..20 | % { curl -X POST http://localhost:5000/run_round; Start-Sleep -Seconds 1 }
```

2. **Viewing Results via Client Script**:

We provided a Python script `(client_summary.py)` to fetch data from the API and
display a professional summary table.

- Ensure the server (port 5000) is running.

- Run the client script in a separate terminal:

```bash
python client_summary.py
```

- Output: It will display a Markdown-formatted table of the training history and
  save a `*.csv` file.

3. **Monitoring via Web Dashboard**:

Open your browser and navigate to:
[http://localhost:5000/dashboard](http://localhost:5000/dashboard)

The dashboard automatically updatess every 2 seconds to show the latest training
statistics.

## Configuration

To adjust simulation parameters (e.g., number of workers, learning rate, noise
epsilon), modify the initialization section in main.py or the class definitions
in app/core/worker.py.

```bash
# Example in main.py
workers = [WorkerNode(i, None) for i in range(5)] # Change number of workers here
```
