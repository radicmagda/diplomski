# Base image with Python + CUDA + PyTorch
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel

# Install required dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgl1 \
    unzip && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Copy and install Python dependencies
COPY requirements.txt . 
RUN pip install --no-cache-dir -r requirements.txt

#--------------- uncomment if in need of regular bash docker shell
# Default command: open bash shell
#CMD ["/bin/bash"]

# Start the Python training script by default
CMD ["python", "basicsr/train.py", "-opt", "options/train/GoPro/NAFNet-width32-v2.yml"]