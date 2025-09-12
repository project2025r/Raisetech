FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (including those for OpenCV and AVIF)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg \
    libavif-dev \
    libavif-bin \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=app.py
ENV FLASK_ENV=production
ENV MONGO_URI=mongodb://mongo:27017
ENV DB_NAME=LTA

# Expose the port that the app runs on
EXPOSE 5000

# Command to run the application
CMD ["python", "app.py"] 