# Use official Python base image
FROM python:3.11-slim

# Install dependencies for Chrome and other essentials
RUN apt-get update && apt-get install -y \
    ffmpeg \
    wget \
    gnupg2 \
    unzip \
    curl \
    ca-certificates \
    fonts-liberation \
    libappindicator3-1 \
    libasound2 \
    libatk-bridge2.0-0 \
    libatk1.0-0 \
    libcups2 \
    libdbus-1-3 \
    libdrm2 \
    libgbm1 \
    libnspr4 \
    libnss3 \
    libx11-xcb1 \
    libxcomposite1 \
    libxdamage1 \
    libxrandr2 \
    xdg-utils \
    --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

# Install Google Chrome stable
RUN wget -q -O - https://dl.google.com/linux/linux_signing_key.pub | apt-key add - && \
    echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" > /etc/apt/sources.list.d/google-chrome.list && \
    apt-get update && apt-get install -y google-chrome-stable && \
    rm -rf /var/lib/apt/lists/*

# Install ChromeDriver via webdriver-manager Python package (optional but recommended)
RUN pip install --no-cache-dir webdriver-manager

# Set working directory
WORKDIR /app

# Copy your requirements.txt and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app source code
COPY . .

# Expose port your Flask app will run on
EXPOSE 5000

# Set environment variable for headless Chrome usage (optional)
ENV CHROME_BIN=/usr/bin/google-chrome

# Command to run your app (adjust if your entrypoint script or filename differs)
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "features_api:app", "--workers=1", "--threads=4"]
