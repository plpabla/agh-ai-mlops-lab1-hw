FROM python:3.12.8-slim-bookworm
RUN pip install uv
COPY pyproject.toml .
RUN uv sync
COPY . .
CMD ["./.venv/bin/python", "-m", "uvicorn", "app:app", "--host", "0.0.0", "--port", "80"]