# FROM nvidia/cudagl:11.4.2-devel-ubuntu20.04

# ENV DEBIAN_FRONTEND=noninteractive

# ENV ROS_DISTRO=noetic
# ENV ROS_ROOT=/opt/ros/noetic
# ENV ROS_PACKAGE_PATH=$ROS_ROOT/share
# ENV PATH=$ROS_ROOT/bin:$PATH
# ENV PYTHONPATH=$ROS_ROOT/lib/python3/dist-packages:$PYTHONPATH
# ENV LD_LIBRARY_PATH=$ROS_ROOT/lib:$LD_LIBRARY_PATH        




# RUN apt-get update && apt-get install -y --no-install-recommends \
#     curl \
#     gnupg2 \
#     lsb-release \
#     software-properties-common






# RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list' && \
#     curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -





# RUN apt-get update && apt-get install -y --no-install-recommends \
#     ros-noetic-desktop-full \
#     python3-pip \
#     python3-colcon-common-extensions \
#     ros-noetic-cv-bridge \
#     ros-noetic-sensor-msgs \
#     ros-noetic-std-msgs \
#     build-essential

       

# RUN pip3 install --upgrade pip
# # RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu114
# # RUN pip3 install rosdep
# RUN pip3 install --no-cache-dir --ignore-installed torch torchvision numpy pillow matplotlib ultralytics filterpy pandas



# RUN mkdir -p /catkin_ws/src

# WORKDIR /catkin_ws

# # Initialize the catkin workspace
# RUN /bin/bash -c "source /opt/ros/noetic/setup.bash && catkin_make"

# # Ensure ROS environment is sourced on container startup
# RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
# RUN echo "source /catkin_ws/devel/setup.bash" >> ~/.bashrc

# # Default working directory
# WORKDIR /catkin_ws/src/yolov5

# #ENTRYPOINT ["rosrun", "yolov5", "node_yolo.py"]

# ROS Noetic on Ubuntu 20.04 (focal)
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
