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


# FROM ros:noetic

# # Environment variables for NVIDIA and non-interactive apt
# ENV NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-all}
# ENV NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics
# ENV DEBIAN_FRONTEND=noninteractive

# # -----------------------------
# # Base utilities
# # -----------------------------
# RUN apt-get update && apt-get install -y \
#     build-essential \
#     curl \
#     ca-certificates \
#     bzip2 \
#     eog \
#     x11-apps \
#     software-properties-common \
#     ros-noetic-tf \
#     && rm -rf /var/lib/apt/lists/*

# # -----------------------------
# # Python 3.10 (via Mambaforge)
# # -----------------------------
# RUN set -eux; \
#     url="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"; \
#     curl -fL "$url" -o /tmp/miniforge.sh; \
#     bash /tmp/miniforge.sh -b -p /opt/conda; \
#     rm /tmp/miniforge.sh

# ENV PATH=/opt/conda/bin:$PATH

# RUN conda install -y -n base -c conda-forge mamba && \
#     conda clean -afy
    


# # Make conda available
# ENV PATH=/opt/conda/bin:$PATH

# # Create a Python 3.10 env and install Python deps there
# RUN mamba create -y -n py310 python=3.10 pip \
#     && mamba run -n py310 pip install --upgrade pip setuptools wheel \
#     && mamba run -n py310 pip install extensisq jax jaxlib pandas

# # Make the py310 env the default Python/pip in this container shell
# ENV CONDA_DEFAULT_ENV=py310
# ENV PATH=/opt/conda/envs/py310/bin:/opt/conda/bin:$PATH

# # -----------------------------
# # Copy your source and install it
# # -----------------------------
# COPY jax_guam /jax_guam
# WORKDIR /jax_guam

# # Install your package into the Python 3.10 env
# RUN pip install -e .

# # -----------------------------
# # ROS environment (keep Noetic settings)
# # -----------------------------
# ENV ROS_DISTRO=noetic
# ENV ROS_ROOT=/opt/ros/noetic
# ENV ROS_PACKAGE_PATH=$ROS_ROOT/share
# ENV PATH=$ROS_ROOT/bin:$PATH
# ENV PYTHONPATH=$ROS_ROOT/lib/python3/dist-packages:$PYTHONPATH
# ENV LD_LIBRARY_PATH=$ROS_ROOT/lib:$LD_LIBRARY_PATH

# RUN mkdir -p /catkin_ws/src

# # Default shell
# CMD ["/bin/bash"]

#FROM ros:noetic
# ROS Noetic on Ubuntu 20.04 (focal)
FROM osrf/ros:noetic-desktop-focal

ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

# Keep PATH "ROS-safe" so catkin uses system python3 (3.8), not conda
ENV PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

# --- Base tools + ROS/catkin deps (system python side) ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl gnupg2 lsb-release ca-certificates \
    software-properties-common \
    build-essential \
    wget bzip2 \
    eog \
    x11-apps \
    ros-noetic-tf \
    python3-empy \
    python3-catkin-pkg \
    python3-rospkg \
    python3-yaml \
    python3-numpy \
 && rm -rf /var/lib/apt/lists/*

# ============================================================
# Miniconda (install it, but DO NOT prepend to PATH)
# ============================================================
ENV CONDA_DIR=/opt/conda

RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
 && bash /tmp/miniconda.sh -b -p ${CONDA_DIR} \
 && rm -f /tmp/miniconda.sh

# Accept Anaconda channel ToS (needed in many CI/Docker envs)
RUN ${CONDA_DIR}/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
 && ${CONDA_DIR}/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

RUN ${CONDA_DIR}/bin/conda config --set always_yes yes --set changeps1 no \
 && ${CONDA_DIR}/bin/conda update -q conda

# Create Python 3.10 env + install:
# - JAX stack + numpy/pandas
# - ROS-with-conda glue deps needed when importing rospy (pyyaml, rospkg, catkin_pkg)
RUN ${CONDA_DIR}/bin/conda create -n py310 python=3.10 \
 && ${CONDA_DIR}/bin/conda run -n py310 python -m pip install --upgrade pip \
 && ${CONDA_DIR}/bin/conda run -n py310 python -m pip install \
      numpy pandas extensisq jax jaxlib \
      pyyaml rospkg catkin_pkg

# ------------------------------------------------------------
# Interpreter wrapper for rosrun scripts:
#   #!/usr/local/bin/py310ros
# ------------------------------------------------------------
RUN printf '%s\n' \
'#!/usr/bin/env bash' \
'set -e' \
'source /opt/ros/noetic/setup.bash' \
'exec /opt/conda/bin/conda run -n py310 python "$@"' \
> /usr/local/bin/py310ros \
 && chmod +x /usr/local/bin/py310ros

# ------------------------------------------------------------
# Entrypoint wrapper:
# At container start (after volumes are mounted), patch scripts so rosrun
# uses py310ros instead of /usr/bin/python3, and make them executable.
# ------------------------------------------------------------
RUN printf '%s\n' \
'#!/usr/bin/env bash' \
'set -e' \
'' \
'SCRIPTDIR="/catkin_ws/src/jaxguam/script"' \
'if [ -d "$SCRIPTDIR" ]; then' \
'  for f in "$SCRIPTDIR"/*.py; do' \
'    [ -f "$f" ] || continue' \
'    first="$(head -n 1 "$f" || true)"' \
'    case "$first" in' \
'      "#! /usr/bin/python3"|"#!/usr/bin/python3"|"#!/usr/bin/env python3")' \
'        sed -i "1c\\#!/usr/local/bin/py310ros" "$f"' \
'        chmod +x "$f" || true' \
'        ;;' \
'    esac' \
'  done' \
'fi' \
'' \
'exec /ros_entrypoint.sh "$@"' \
> /ros_entrypoint_py310.sh \
 && chmod +x /ros_entrypoint_py310.sh

# (Optional) install your python package into py310 if you use "import jax_guam"
COPY jax_guam /jax_guam
WORKDIR /jax_guam
RUN ${CONDA_DIR}/bin/conda run -n py310 python -m pip install -e .

# ROS env vars
ENV ROS_DISTRO=noetic
ENV ROS_ROOT=/opt/ros/noetic
ENV ROS_PACKAGE_PATH=$ROS_ROOT/share
ENV PYTHONPATH=$ROS_ROOT/lib/python3/dist-packages:$PYTHONPATH
ENV LD_LIBRARY_PATH=$ROS_ROOT/lib:$LD_LIBRARY_PATH

RUN mkdir -p /catkin_ws/src
WORKDIR /catkin_ws

ENTRYPOINT ["/ros_entrypoint_py310.sh"]
CMD ["bash"]
