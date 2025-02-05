#!/bin/bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python3 src/load_data.py
python3 src/train_model.py
python3 src/analyze_model_validation.py
python3 src/analyze_model_drift.py
python3 src/analyze_target_drift.py
python3 src/analyze_data_drift.py    

evidently ui --workspace ./datascientest-workspace/ --host 0.0.0.0 --port 8888