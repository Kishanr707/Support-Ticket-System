# Docker Guide

## Building and running just the backend

```bash
docker build -t ticket-priority-backend .
docker run -p 8000:8000 ticket-priority-backend
```

Then test it the same way as before:
- `http://localhost:8000/health`
- `http://localhost:8000/docs`

## Using docker-compose (recommended once you're testing regularly)

```bash
docker-compose up --build
```

This builds and runs the backend the same way, just with one command
instead of two, and sets things up so the frontend service can be
uncommented and added later without restructuring anything.

## Important: rebuilding after retraining the model

The Docker image bakes in whatever `models/priority_model.joblib`
exists at build time. If you retrain the model (e.g. after adding new
data), you must rebuild the image for the container to use the new
model — it won't pick up a retrained model automatically:

```bash
docker build -t ticket-priority-backend .
```

## For your friend, once his frontend has a Dockerfile

1. Uncomment the `frontend:` service block in `docker-compose.yml`
2. Set `context:` to wherever his repo is cloned relative to this file
   (simplest: clone both repos into the same parent folder, e.g.:
   ```
   projects/
   ├── Support Ticket System/     (this repo)
   └── frontend-repo/             (his repo)
   ```
   then `context: ../frontend-repo` is correct as-is)
3. **Networking**: once both run via `docker-compose up`, the frontend
   container must call `http://backend:8000/predict` — not
   `http://localhost:8000/predict`. Containers on the same
   docker-compose network address each other by service name.
   `localhost` inside a container only refers to itself.
4. Run `docker-compose up --build` from this repo's root — it builds
   and starts both containers together.

## What's NOT included in the image (on purpose)

- `data/` (raw training CSVs — large, only needed for retraining, not serving)
- `tests/`, `notebooks/` — not needed to run the API

If you need to retrain, do it locally (outside Docker) as before with
`python -m src.model.train`, then rebuild the image so the new model
gets baked in.