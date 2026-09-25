from src.model.predict import load_model
from src.preprocessing.clean import clean_text

tickets = [
    "VPN is completely down for the entire office",
    "Please reset my password",
    "The printer is not working",
    "The application is slightly slow",
    "The production server is completely down and all users are affected",
    "I need a new mouse",
    "Database is unavailable and customers cannot access the system",
]

model = load_model()

for ticket in tickets:
    cleaned = clean_text(ticket)
    probabilities = model.predict_proba([cleaned])[0]

    print(f"\nTicket: {ticket}")

    for label, probability in sorted(
        zip(model.classes_, probabilities),
        key=lambda x: x[1],
        reverse=True,
    ):
        print(f"  {label:8} {probability:.3f}")