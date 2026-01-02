# qhd-simulations

Reproducible Python code for numerical experiments related to Quantum Hamiltonian Descent (QHD),
developed in the context of an MSc thesis project.

## Repository layout
- `src/MAIN.py`: main entry point (runs simulations / experiments)
- `src/QHD_quantum.py`: core routines (QHD-related implementation)
- `src/`: additional modules used by the main script

## Requirements
- Python 3.10+ recommended
- See `requirements.txt` (to be added)

## Quickstart (local)
Create and activate a virtual environment, then run the main script:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/MAIN.py
