# Base image with Python + CUDA + PyTorch
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel

# Install system dependencies, including glib
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*
# Set working directory inside the container
WORKDIR /workspace

# Copy requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Default command: open bash shell
CMD ["/bin/bash"]