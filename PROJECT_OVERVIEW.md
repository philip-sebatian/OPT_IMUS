# workspace_optimus Overview

workspace_optimus is a packaging of the Optimus routing toolkit focused on rapid experimentation and live visualisation. The repository couples NVIDIA cuOpt powered solvers with modern Folium/Streamlit front-ends.

## Directory Map

```
workspace_optimus/
├── dashboard/                # Streamlit dashboard (map + graph views)
├── visualization/            # Folium and canvas visualisation helpers
├── src/                      # Core optimisation library
│   ├── algorithms/           # cuOpt integration + heuristics
│   ├── core/                 # Vehicles, depot manager, tasks, costs
│   └── utils/                # Graph utilities and pretty printers
├── examples/                 # Scriptable demos
├── tests/                    # Unit/integration tests
├── docs/                     # API & planner deep dives
├── requirements.txt          # Dependencies
├── setup.py                  # Packaging metadata
└── README.md                 # Full documentation
```

## Quick Start

```bash
cd workspace_optimus
pip install -r requirements.txt
pip install -e .

# Launch the dashboard
export PYTHONPATH=$(pwd)
streamlit run workspace_optimus/dashboard/app.py --server.port=8501
```

Open `http://0.0.0.0:8501` and switch between **Map view** (Folium) and **Graph view** (canvas animation) to inspect planners.

## Tests

```bash
pytest
```

## Support

Please report issues or ideas through your preferred tracker referencing the workspace_optimus build.
