FROM quay.io/jupyter/minimal-notebook:latest

# Build-time argument for requirements path - defaults for building from jupyter dir
ARG REQUIREMENTS_PATH=requirements.txt

USER root
RUN apt-get update && \
    apt-get install -y default-jdk ca-certificates-java libopenblas-dev \
    build-essential \
    tesseract-ocr libtesseract-dev poppler-utils \
    libgeos-dev libproj-dev proj-data proj-bin \
    libgtk2.0-dev libgstreamer-plugins-base1.0-dev \
    libgl1 libgomp1 libglib2.0-0 \
    curl sqlite3 libsqlite3-dev gcc && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    update-ca-certificates -f

    # Install R and IRkernel
# RUN apt-get update && \
#     apt-get install -y --no-install-recommends r-base \
#     r-base-dev\
#     libcurl4-openssl-dev \
#     libssl-dev \
#     libxml2-dev && \
#     apt-get clean && \
#     rm -rf /var/lib/apt/lists/* && \
#     R -e "install.packages('IRkernel', repos='http://cran.rstudio.com/')" && \
#     R -e "IRkernel::installspec(user = FALSE)"

# Set JAVA_HOME environment variable 
ENV JAVA_HOME=/usr/lib/jvm/java-21-openjdk-arm64
# Set PATH environment variable
ENV PATH=$JAVA_HOME/bin:$PATH

# Copy and install requirements as root first
COPY ${REQUIREMENTS_PATH} /tmp/requirements.txt

# Verify requirements file was copied and show first few lines
RUN echo "=== Verifying requirements.txt ===" && \
    ls -la /tmp/requirements.txt && \
    head -10 /tmp/requirements.txt || (echo "ERROR: requirements.txt not found" && exit 1)

# Install Python packages
# The Jupyter minimal notebook base image uses conda, but pip should work
RUN echo "=== Installing Python packages ===" && \
    python --version && \
    pip --version && \
    pip install --no-cache-dir --upgrade pip setuptools wheel && \
    echo "=== Installing from requirements.txt ===" && \
    pip install --no-cache-dir -r /tmp/requirements.txt && \
    echo "=== Installing spaCy models ===" && \
    python -m spacy download en_core_web_sm && \
    python -m spacy download en_core_web_lg && \
    echo "=== Verifying pandas installation ===" && \
    python -c "import pandas; print(f'✓ pandas {pandas.__version__} installed successfully')" && \
    pip list | grep -E "(pandas|numpy|matplotlib)" || \
    (echo "ERROR: Critical packages missing!" && pip list && exit 1)

# Switch to non-root user after installation
USER ${NB_UID}
