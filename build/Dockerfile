# Use an official CUDA base image from NVIDIA
FROM nvidia/cuda:12.1.0-base-ubuntu22.04

# Set environment variables
ENV PATH /opt/miniforge/bin:$PATH
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

# Install Miniforge
RUN curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh" && \
    bash Miniforge3-$(uname)-$(uname -m).sh -b -p /opt/miniforge && \
    rm Miniforge3-$(uname)-$(uname -m).sh

# Copy your environment.yml into the Docker image
COPY environment.yml /opt/environment.yml

# Create the Conda environment using Miniforge
RUN /opt/miniforge/bin/conda env create -f /opt/environment.yml

# Clean up conda packages to reduce the container size
RUN /opt/miniforge/bin/conda clean -a --yes

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "dronalize", "/bin/bash", "-c"]

# The code to run when container is started
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "dronalize"]
CMD ["python", "--version"]
