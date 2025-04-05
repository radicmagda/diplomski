# Base image with Python + CUDA + PyTorch
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel

# Install required dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libglib2.0-dev \
    libsm6 \
    libxext6 \
    libxrender1 \
    unzip && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Copy and install Python dependencies
COPY requirements.txt . 
RUN pip install --no-cache-dir -r requirements.txt

# Default command: open bash shell
CMD ["/bin/bash"]