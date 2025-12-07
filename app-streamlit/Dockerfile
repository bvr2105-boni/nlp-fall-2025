FROM python:3.11-slim

WORKDIR /usr/src/app

# Build-time arguments for requirements paths - defaults for building from app-streamlit dir
ARG REQUIREMENTS_PATH=requirements.txt
ARG REQUIREMENTS_DEV_PATH=requirements-dev.txt

# dont write pyc files
# dont buffer to stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY ${REQUIREMENTS_PATH} /usr/src/app/requirements.txt
COPY ${REQUIREMENTS_DEV_PATH} /usr/src/app/requirements-dev.txt

# System dependencies (graphics/libs needed by various Python packages)
RUN apt-get update && apt-get install -y libgl1 libgomp1 libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Additional tools and libraries (OCR, GIS, build tools, etc.)
RUN apt-get update && apt-get -y install --no-install-recommends \
    curl sqlite3 libsqlite3-dev gcc build-essential \
    tesseract-ocr libtesseract-dev poppler-utils \
    libgeos-dev libproj-dev proj-data proj-bin \
    libgtk2.0-dev libgstreamer-plugins-base1.0-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Chromium browser and Chromedriver for Selenium / linkedin-jobs-scraper
RUN apt-get update && apt-get -y install --no-install-recommends \
    chromium chromium-driver \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Default paths used by 0_Job_Crawling.py (can be overridden via env vars)
ENV CHROME_BINARY=/usr/bin/chromium
ENV CHROMEDRIVER_PATH=/usr/bin/chromedriver

RUN pip install --upgrade pip setuptools wheel \
    && pip3 install --only-binary h5py h5py \
    && pip install -r requirements.txt \
    && pip install -r requirements-dev.txt \
    && python -m spacy download en_core_web_sm \
    && python -m spacy download en_core_web_lg || echo "Note: en_core_web_lg is optional (large model)" \
    && rm -rf /root/.cache/pip

