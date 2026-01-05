FROM osrf/ros:noetic-desktop-focal

ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics
ENV PATH="/usr/local/bin:${PATH}"

# --- Base tools ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl gnupg2 lsb-release ca-certificates \
    software-properties-common \
    build-essential \
    wget bzip2 \
    eog \
    x11-apps \
    ros-noetic-tf \
 && rm -rf /var/lib/apt/lists/*

# --- ROS key ---
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    | gpg --dearmor -o /usr/share/keyrings/ros-archive-keyring.gpg

RUN echo "deb [arch=amd64 signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" \
    > /etc/apt/sources.list.d/ros1.list

# ============================================================
# Python 3.10 via Miniconda
# ============================================================
ENV CONDA_DIR=/opt/conda
ENV PATH=${CONDA_DIR}/bin:$PATH

RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
 && bash /tmp/miniconda.sh -b -p ${CONDA_DIR} \
 && rm -f /tmp/miniconda.sh \
 && conda config --set always_yes yes --set changeps1 no \
 && conda update -q conda

RUN conda create -n py310 python=3.10 \
 && conda run -n py310 python -m pip install --upgrade pip \
 && conda run -n py310 python -m pip install extensisq jax jaxlib pandas

# Copy and install jax_guam
COPY jax_guam /jax_guam
WORKDIR /jax_guam
RUN conda run -n py310 python -m pip install -e .

# --- ROS environment vars ---
ENV ROS_DISTRO=noetic
ENV ROS_ROOT=/opt/ros/noetic
ENV ROS_PACKAGE_PATH=$ROS_ROOT/share
ENV PATH=$ROS_ROOT/bin:$PATH
ENV PYTHONPATH=$ROS_ROOT/lib/python3/dist-packages:$PYTHONPATH
ENV LD_LIBRARY_PATH=$ROS_ROOT/lib:$LD_LIBRARY_PATH

RUN mkdir -p /catkin_ws/src
WORKDIR /catkin_ws

CMD ["/bin/bash"]
