"""
src/model/predict.py

Stage 5: the stable prediction interface.

predict(ticket_text) is the ONE function everything downstream calls —
tests now, FastAPI's /predict endpoint later. As long as this signature
stays the same, the model behind it (LogisticRegression today, maybe
something else later) can change without anything else in the project
needing to change.

Important: raw ticket text must go through the same clean_text() used
during training before being handed to the model. Skipping this step
would mean predicting on differently-formatted text than the model was
trained on, silently degrading accuracy.
"""

from functools import lru_cache

import joblib

from src.preprocessing.clean import clean_text

DEFAULT_MODEL_PATH = "models/priority_model.joblib"


@lru_cache(maxsize=None)
def load_model(model_path: str = DEFAULT_MODEL_PATH):
    """Load and cache the trained pipeline.

    Cached with lru_cache so repeated calls (e.g. many requests in a
    running API) don't re-read the model file from disk each time —
    it's loaded once per process.

    Args:
        model_path: path to the saved joblib pipeline

    Returns:
        Fitted sklearn Pipeline (TF-IDF + classifier)
    """
    return joblib.load(model_path)


def predict(ticket_text: str, model_path: str = DEFAULT_MODEL_PATH) -> str:
    """Predict the priority of a single ticket from raw text.

    Args:
        ticket_text: raw, unprocessed ticket text (e.g. straight from a
            web form or API request)
        model_path: path to the saved joblib pipeline

    Returns:
        Predicted priority label, e.g. "High"
    """
    model = load_model(model_path)
    cleaned = clean_text(ticket_text)
    return model.predict([cleaned])[0]


if __name__ == "__main__":
    sample = "The VPN keeps disconnecting every few minutes, this is blocking my whole team from working."
    result = predict(sample)
    print(f"Ticket: {sample}")
    print(f"Predicted priority: {result}")