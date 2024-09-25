# Use an official CUDA base image from NVIDIA
FROM nvidia/cuda:12.1.0-base-ubuntu22.04

# Set environment variables
ENV PATH /opt/mambaforge/bin:$PATH
ENV PYTHONNOUSERSITE 1

# Run updates and install necessary packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    nano \
    wget \
    curl \
    ca-certificates && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Mambaforge
RUN curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh" && \
    bash Mambaforge-$(uname)-$(uname -m).sh -b -p /opt/mambaforge && \
    rm Mambaforge-$(uname)-$(uname -m).sh

# Copy your environment.yml into the Docker image
COPY environment.yml /opt/environment.yml

# Create the Conda environment using Mamba
RUN /opt/mambaforge/bin/mamba env create -f /opt/environment.yml

# Clean up conda packages to reduce the container size
RUN /opt/mambaforge/bin/mamba clean -a --yes

# Make RUN commands use the new environment:
SHELL ["mamba", "run", "-n", "dronalize", "/bin/bash", "-c"]

# The code to run when container is started
ENTRYPOINT ["mamba", "run", "--no-capture-output", "-n", "dronalize"]
CMD ["python", "--version"]
