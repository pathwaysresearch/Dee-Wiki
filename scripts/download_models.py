"""
Download ONNX embedding models into webapp/models/ so they are committed to the repo
and available on Vercel without a runtime download.

Usage:
    python scripts/download_models.py
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "webapp" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",
]

if __name__ == "__main__":
    from fastembed import TextEmbedding

    for model_name in MODELS:
        print(f"Downloading {model_name} → {MODELS_DIR}")
        model = TextEmbedding(model_name, cache_dir=str(MODELS_DIR))
        list(model.embed(["warmup"]))
        print(f"  Done — {model_name}")

    print(f"\nAll models saved to {MODELS_DIR}")
    print("Commit webapp/models/ to git before deploying to Vercel.")
