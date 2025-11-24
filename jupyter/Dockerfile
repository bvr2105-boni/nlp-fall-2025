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
RUN pip install -r /tmp/requirements.txt

# Switch to non-root user after installation
USER ${NB_UID}
