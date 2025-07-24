FROM python:3.10-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    chromium-driver \
    libglib2.0-0 libnss3 libgconf-2-4 libfontconfig1 libxss1 libappindicator3-1 libasound2 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Runtime environment variables
ENV PORT=8080

CMD ["python", "app.py"]