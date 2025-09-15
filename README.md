Starter repository for Deployment Lab: DVC + MLflow + FastAPI + Docker + GitHub Actions

Structure:
- data/data.csv            : sample dataset
- train.py                 : trains a simple model, logs to MLflow, and saves model.pkl
- app.py                   : FastAPI app that loads model.pkl and serves /predict
- requirements.txt         : Python dependencies
- Dockerfile               : containerize the FastAPI app
- docker-compose.yml       : optional compose to run the app locally
- .github/workflows/ci.yml : GitHub Actions stub to run tests and build image
- tests/test_api.py        : basic test that runs training then checks the API
- .gitignore               : ignore common files
- dvc_files/data.dvc       : placeholder .dvc pointer for dataset (students will run dvc add themselves)

Usage (local):
1. Create a virtual env and install requirements.
2. Run `python train.py` to train and produce model.pkl (and logs in mlruns/).
3. Run `uvicorn app:app --reload --port 8000` to serve the API.
4. Test via POST to /predict with JSON: { "features": [1.0] }
