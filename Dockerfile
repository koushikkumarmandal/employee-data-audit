# Python environment
FROM python:3.11-slim

# Project folder inside Docker
WORKDIR /app

# Install required libraries
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project code, packages, notebooks, configuration, reports, and tests
COPY pyproject.toml .
COPY src/ ./src/
COPY cortexhr_features/ ./cortexhr_features/
COPY notebooks/ ./notebooks/
COPY tests/ ./tests/
COPY reports/ ./reports/
COPY models_store/ ./models_store/
COPY data/raw/ ./data/raw/

# Create folders for generated results
RUN mkdir -p \
    /app/data/processed \
    /app/reports \
    /app/models

# JupyterLab port
EXPOSE 8888

# Start JupyterLab by default
# (You can also run tests: docker run <image> pytest)
# (Or run pipeline: docker run -v ${PWD}/data:/app/data <image> python -m src.features.build_features)
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--ServerApp.token=student"]
