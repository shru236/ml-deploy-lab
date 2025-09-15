import subprocess, os
from fastapi.testclient import TestClient
import app as app_module

client = TestClient(app_module.app)

def test_health_and_predict():
    # Ensure model exists; if not, create it via train.py
    if not os.path.exists('model.pkl'):
        subprocess.run(['python', 'train.py'], check=True)
    # Health endpoint
    r = client.get('/health')
    assert r.status_code == 200 and r.json().get('status') == 'ok'
    # Predict endpoint (use x=6 -> expected approx 600 based on linear relation)
    r = client.post('/predict', json={'features': [6]})
    assert r.status_code == 200
    assert 'prediction' in r.json()
