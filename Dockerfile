# Base image with Python + CUDA + PyTorch
FROM pytorch/pytorch:1.13.1-cuda11.7-cudnn8-runtime

# Set working directory inside the container
WORKDIR /workspace

# Copy requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Default command: open bash shell
CMD ["/bin/bash"]