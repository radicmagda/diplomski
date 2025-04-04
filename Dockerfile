# Base image with Python + CUDA + PyTorch
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# Set working directory inside the container
WORKDIR /workspace

# Copy requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Default command: open bash shell
CMD ["/bin/bash"]