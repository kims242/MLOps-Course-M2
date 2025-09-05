PY=python
ENV?=.venv
EXP?=churn-exp

init:
	python -m venv $(ENV) && . $(ENV)/bin/activate && pip install -U pip -r requirements.txt

fetch:
	$(PY) src/utils.py --fetch  # ou créer une fonction fetch_dataset dans utils

train:
	MLFLOW_EXPERIMENT_NAME=$(EXP) PYTHONPATH=. $(PY) /home/user/MLOps-Course-M2/project/src/train.py --config configs/config.yaml

evaluate:
	MLFLOW_EXPERIMENT_NAME=$(EXP) PYTHONPATH=. $(PY) src/evaluate.py --config configs/config.yaml

test:
	PYTHONPATH=. $(PY) -m pytest -q

## pour visualiser les expériences
# mlflow ui --backend-store-uri ./mlruns --port 5000 
# ==> puis aller sur http://localhost:5000
