# api/model_fetch.py
import os
import zipfile
import requests

def ensure_models(model_dir: str, model_zip_url: str):
    os.makedirs(model_dir, exist_ok=True)

    required = ["vectorizer.joblib", "estimator.joblib", "mlb.joblib", "config.json"]
    if all(os.path.exists(os.path.join(model_dir, f)) for f in required):
        return

    if not model_zip_url:
        raise RuntimeError("MODEL_BLOB_URL is not set and models are missing.")

    zip_path = os.path.join(model_dir, "model_artifacts.zip")

    r = requests.get(model_zip_url, timeout=120)
    r.raise_for_status()
    with open(zip_path, "wb") as f:
        f.write(r.content)

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(model_dir)

    # Optionnel: cleanup
    try:
        os.remove(zip_path)
    except OSError:
        pass