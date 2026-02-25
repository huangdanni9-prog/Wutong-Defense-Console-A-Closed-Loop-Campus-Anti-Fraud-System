# syntax=docker/dockerfile:1

# Wutong Defense Console - Streamlit Frontend
ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}-slim as base

# Prevents Python from writing pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Fix matplotlib config directory
ENV MPLCONFIGDIR=/tmp/matplotlib

# Set PYTHONPATH so imports work correctly
ENV PYTHONPATH=/app/src:/app/src/frontend

# Groq API Key - pass at runtime: docker run -e GROQ_API_KEY=your_key ...
ENV GROQ_API_KEY=""

WORKDIR /app

# Install dependencies
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=requirements.txt,target=requirements.txt \
    python -m pip install -r requirements.txt

# Copy the source code into the container
COPY . .

# Create necessary directories and set permissions
RUN mkdir -p /app/Datasets/Student/Results \
    /app/Datasets/Fraud/Results \
    /app/models \
    /tmp/matplotlib && \
    chmod -R 777 /app/Datasets /app/models /tmp/matplotlib

# Expose Streamlit default port
EXPOSE 8501

# Healthcheck for container orchestration
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run Streamlit frontend
# --server.address=0.0.0.0 allows external connections
# --server.headless=true for containerized environments
CMD ["python", "-m", "streamlit", "run", "src/frontend/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
