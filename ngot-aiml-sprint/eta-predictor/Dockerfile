FROM python:3.12-slim

WORKDIR /app

# Install Poetry
RUN pip install --no-cache-dir poetry

# Copy dependency files
COPY pyproject.toml poetry.lock* ./

# Install dependencies
RUN poetry config virtualenvs.create false && \
    poetry install --no-interaction --no-ansi

# Copy source code
COPY src ./src

# Expose port for FastAPI
EXPOSE 8000

# Default command: run FastAPI with uvicorn
CMD ["uvicorn", "eta_predictor.main:app", "--host", "0.0.0.0", "--port", "8000"]
