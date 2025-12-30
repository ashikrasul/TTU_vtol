# FROM ros:noetic

# # Set environment variables for NVIDIA and non-interactive apt
# ENV NVIDIA_VISIBLE_DEVICES ${NVIDIA_VISIBLE_DEVICES:-all}
# ENV NVIDIA_DRIVER_CAPABILITIES ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics
# ENV DEBIAN_FRONTEND=noninteractive
# ENV PATH="/usr/local/bin:${PATH}"

# # Update and install essential tools in one layer to minimize image size
# RUN apt-get update && apt-get install -y \
#     software-properties-common \
#     build-essential \
#     curl \
#     eog \
#     x11-apps \
#     ros-noetic-tf \
#     && rm -rf /var/lib/apt/lists/*

# RUN apt-get update && apt-get upgrade -y && rm -rf /var/lib/apt/lists/*

# # Install Python 3.10 and set it as the default
# RUN add-apt-repository ppa:deadsnakes/ppa && \
#     apt-get update && apt-get install -y python3.10 python3.10-distutils python3.10-venv && \
#     update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
#     curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10 && \
#     rm -rf /var/lib/apt/lists/*

# # Install Python dependencies for JAX and other tools
# RUN python3.10 -m pip install --upgrade pip && \
#     python3.10 -m pip install extensisq jax jaxlib pandas

# # Copy the jax_guam source and install it
# COPY jax_guam /jax_guam
# WORKDIR /jax_guam
# RUN python3.10 -m pip install -e .

# # Set up ROS environment
# ENV ROS_DISTRO=noetic
# ENV ROS_ROOT=/opt/ros/noetic
# ENV ROS_PACKAGE_PATH=$ROS_ROOT/share
# ENV PATH=$ROS_ROOT/bin:$PATH
# ENV PYTHONPATH=$ROS_ROOT/lib/python3/dist-packages:$PYTHONPATH
# ENV LD_LIBRARY_PATH=$ROS_ROOT/lib:$LD_LIBRARY_PATH

# RUN mkdir -p /catkin_ws/src

#FROM ros:noetic
FROM ros:noetic

# Environment variables for NVIDIA and non-interactive apt
ENV NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics
ENV DEBIAN_FRONTEND=noninteractive

# -----------------------------
# Base utilities
# -----------------------------
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    ca-certificates \
    bzip2 \
    eog \
    x11-apps \
    software-properties-common \
    ros-noetic-tf \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------
# Python 3.10 (via Mambaforge)
# -----------------------------
RUN set -eux; \
    url="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"; \
    curl -fL "$url" -o /tmp/miniforge.sh; \
    bash /tmp/miniforge.sh -b -p /opt/conda; \
    rm /tmp/miniforge.sh

ENV PATH=/opt/conda/bin:$PATH

RUN conda install -y -n base -c conda-forge mamba && \
    conda clean -afy
    


# Make conda available
ENV PATH=/opt/conda/bin:$PATH

# Create a Python 3.10 env and install Python deps there
RUN mamba create -y -n py310 python=3.10 pip \
    && mamba run -n py310 pip install --upgrade pip setuptools wheel \
    && mamba run -n py310 pip install extensisq jax jaxlib pandas

# Make the py310 env the default Python/pip in this container shell
ENV CONDA_DEFAULT_ENV=py310
ENV PATH=/opt/conda/envs/py310/bin:/opt/conda/bin:$PATH

# -----------------------------
# Copy your source and install it
# -----------------------------
COPY jax_guam /jax_guam
WORKDIR /jax_guam

# Install your package into the Python 3.10 env
RUN pip install -e .

# -----------------------------
# ROS environment (keep Noetic settings)
# -----------------------------
ENV ROS_DISTRO=noetic
ENV ROS_ROOT=/opt/ros/noetic
ENV ROS_PACKAGE_PATH=$ROS_ROOT/share
ENV PATH=$ROS_ROOT/bin:$PATH
ENV PYTHONPATH=$ROS_ROOT/lib/python3/dist-packages:$PYTHONPATH
ENV LD_LIBRARY_PATH=$ROS_ROOT/lib:$LD_LIBRARY_PATH

RUN mkdir -p /catkin_ws/src

# Default shell
CMD ["/bin/bash"]
