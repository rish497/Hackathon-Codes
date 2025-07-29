FROM python:3

# Set working directory
WORKDIR /home/app

# Install system dependencies including tesseract-ocr
RUN apt-get update && \
    apt-get install -y tesseract-ocr libsm6 libxext6 libxrender-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose port 3000
EXPOSE 3000

# Run your app
CMD ["python", "server.py"]
