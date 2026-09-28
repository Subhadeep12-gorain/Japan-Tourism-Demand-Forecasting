FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY deploy/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and models
COPY deploy/app.py .
COPY saved_models/ /app/saved_models/

# Expose port
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
