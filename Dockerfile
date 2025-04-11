# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies and clean up
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy only requirements file first to leverage Docker cache
COPY app/requirements.txt /app/requirements.txt

# Upgrade pip and install dependencies
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY app/ /app/

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
