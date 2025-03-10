# Define build argument for CUDA version (defaults to 12.4)
ARG CUDA_VERSION=12.4

# Select base image for CUDA versions
FROM pytorch/pytorch:2.5.1-cuda${CUDA_VERSION}-cudnn9-runtime

ARG CU_VERSION="124"

# Set working directory
WORKDIR /app

# Copy and install dependencies
COPY container_req.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Install `torch-cluster`
RUN pip install --no-cache-dir torch-cluster -f https://data.pyg.org/whl/torch-2.5.0+cu${CU_VERSION}.html

# Set default command
CMD ["python", "--version"]

# Example build command
# docker build --build-arg CUDA_VERSION=12.4 --build-arg CU_VERSION=124 -t dronalize:latest-cu124 -f Dockerfile .