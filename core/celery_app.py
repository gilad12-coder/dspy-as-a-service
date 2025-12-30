"""Celery application configuration for distributed task processing."""

import os
from pathlib import Path

from celery import Celery
from dotenv import load_dotenv

# Load .env file from project root
load_dotenv(Path(__file__).parent.parent / ".env")

# Redis connection settings - configurable via environment variables
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = os.getenv("REDIS_PORT", "6379")
REDIS_DB_BROKER = os.getenv("REDIS_DB_BROKER", "0")
REDIS_DB_BACKEND = os.getenv("REDIS_DB_BACKEND", "1")

BROKER_URL = os.getenv(
    "CELERY_BROKER_URL",
    f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB_BROKER}",
)
BACKEND_URL = os.getenv(
    "CELERY_RESULT_BACKEND",
    f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB_BACKEND}",
)

celery_app = Celery(
    "dspy_service",
    broker=BROKER_URL,
    backend=BACKEND_URL,
    include=["core.tasks"],
)

celery_app.conf.update(
    # Task tracking
    task_track_started=True,
    task_send_sent_event=True,

    # Serialization
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],

    # Result expiration (24 hours)
    result_expires=86400,

    # Task execution settings
    task_acks_late=True,
    task_reject_on_worker_lost=True,

    # Worker settings
    worker_prefetch_multiplier=1,
    worker_concurrency=int(os.getenv("CELERY_CONCURRENCY", "2")),

    # Timezone
    timezone="UTC",
    enable_utc=True,
)
