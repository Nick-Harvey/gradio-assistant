# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /usr/src/app

# Set environment variables:
# - PYTHONUNBUFFERED: prevents Python from buffering stdout and stderr
# - PYTHONDONTWRITEBYTECODE: prevents Python from writing pyc files to disc
ENV PYTHONUNBUFFERED 1
ENV PYTHONDONTWRITEBYTECODE 1

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /usr/src/app
COPY . .

# Expose the port the app runs on
EXPOSE 7860

# Command to run the app
CMD ["python", "./main.py"]