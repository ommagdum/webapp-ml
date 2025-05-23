# Use official Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system deps (for nltk & joblib)
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
       build-essential \
   && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download NLTK data so your service won't block at runtime
RUN python -m nltk.downloader punkt stopwords

# Create models directory and copy model files
RUN mkdir -p /app/models
COPY models/ /app/models/

# Copy the rest of your application code
COPY . .

# Ensure the models directory has the right permissions
RUN chmod -R a+rwx /app/models

# Expose the port your Flask app runs on
EXPOSE 5001

# Run the Flask application
CMD ["python", "app.py"]
