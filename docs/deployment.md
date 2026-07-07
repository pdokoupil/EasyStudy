# Deployment

EasyStudy is a Flask app. Configuration is entirely via environment variables (see `.env.example`).

## Configuration

| Variable | Default | Notes |
|----------|---------|-------|
| `SECRET_KEY` | random per boot | **Set a fixed value in production** (else sessions reset on restart). |
| `DATABASE_URL` | `sqlite:///db.sqlite` | Use Postgres for real deployments, e.g. `postgresql+psycopg://user:pass@host/db`. |
| `SESSION_TYPE` | `sqlalchemy` | `redis` (+ `REDIS_URL`) for a shared, faster session store. |
| `REDIS_URL` | unset | Only needed if `SESSION_TYPE=redis`. |
| `PORT` | `5000` | HTTP port. |
| `GUNICORN_WORKERS` | `1` | **Keep at 1** unless you've done the shared-state rework (see note). |
| `GUNICORN_TIMEOUT` | `0` | Gunicorn worker timeout (seconds). |
| `EASYSTUDY_EXTRAS` | empty | Build-time: heavy extras baked into the Docker image (`tensorflow,lenskit`). |

## Docker (recommended)

```bash
docker compose run --rm fetch      # first run: fetch datasets
docker compose up --build          # http://localhost:5000
```

- Include heavy backends: `EASYSTUDY_EXTRAS="tensorflow,lenskit" docker compose up --build`.
- Redis sessions: `docker compose --profile redis up` and set `SESSION_TYPE=redis`,
  `REDIS_URL=redis://redis:6379`.

Put a reverse proxy (nginx/Caddy/Traefik) in front for TLS.

## From source (uWSGI/Gunicorn)

```bash
pip install ".[tensorflow]"        # or just "." for the lightweight core
cd server
gunicorn -w 1 --bind 0.0.0.0:5000 "app:create_app()"
```

!!! warning "Concurrency & the single worker"
    Today the per-participant fine-tuned model state is kept **in worker memory**. Running multiple
    Gunicorn workers can therefore send a participant's requests to a worker that doesn't have their
    state. Until the shared-state rework lands (moving fine-tuning to a task queue / storing per-user
    deltas — see the technical backlog), keep `GUNICORN_WORKERS=1` and scale by running independent
    instances behind a load balancer with sticky sessions, or by mirroring studies across servers.

## Database migrations

Schema changes ship as Flask-Migrate migrations:

```bash
cd server
flask db upgrade         # apply
flask db migrate -m "…"  # create a new migration after changing models.py
```

## Backups
Back up the database (`server/instance/db.sqlite` or your Postgres) regularly during a live study — it
holds all collected data.
