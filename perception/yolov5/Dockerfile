FROM nvidia/cudagl:11.4.2-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gnupg2 \
    lsb-release \
    software-properties-common

RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list' && \
    curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -

RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-noetic-desktop-full \
    python3-pip \
    python3-colcon-common-extensions

ENV ROS_DISTRO=noetic
ENV ROS_ROOT=/opt/ros/noetic
ENV ROS_PACKAGE_PATH=$ROS_ROOT/share
ENV PATH=$ROS_ROOT/bin:$PATH
ENV PYTHONPATH=$ROS_ROOT/lib/python3/dist-packages:$PYTHONPATH
ENV LD_LIBRARY_PATH=$ROS_ROOT/lib:$LD_LIBRARY_PATH        

RUN pip3 install --upgrade pip
# RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu114
# RUN pip3 install rosdep

COPY yolov5 /yolov5
WORKDIR /yolov5
RUN pip install -r requirements.txt

RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc

RUN mkdir -p /catkin_ws/src
