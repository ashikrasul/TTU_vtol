FROM nvidia/cudagl:11.4.2-devel-ubuntu20.04

ARG CARLA_VERSION=0.9.15
ARG ROS_VERSION=noetic
ARG GID=1000
ARG UID=1000
ARG USER=sim

ENV NVIDIA_VISIBLE_DEVICES ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/usr/local/bin:${PATH}"

ENV USERNAME ${USER}
ENV HOME /home/$USERNAME
RUN groupadd -f -g ${GID} ${USERNAME} && \
    useradd -m $USERNAME -u ${UID} -g ${GID} && \
    echo "$USERNAME:$USERNAME" | chpasswd && \
    usermod --shell /bin/bash $USERNAME && \
    usermod -aG sudo $USERNAME && \
    mkdir -p /etc/sudoers.d && \
    echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/$USERNAME && \
    chmod 0440 /etc/sudoers.d/$USERNAME && \
    \
    \
    apt-get update && apt-get install -y --no-install-recommends curl software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y --no-install-recommends python3.10-dev python3.10-distutils && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    curl -sSL https://bootstrap.pypa.io/get-pip.py | python3 && \
    ln -s /usr/local/bin/pip /usr/bin/pip3 && \
    pip install --upgrade pip

# Fix some .so paths to accomodate for python 3.10, then install packages
COPY docker/create_links.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/create_links.sh && /usr/local/bin/create_links.sh && \
    \
    \
    sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list' && \
    apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654 && \
    apt-get update && apt-get install -y --no-install-recommends \
        bash-completion \
        build-essential \
        cmake \
        command-not-found \
        freeglut3-dev \
        g++ \
        git \
        less \
        libboost-all-dev \
        libeigen3-dev \
        libomp-dev \
        libopencv-dev \
        libqglviewer-dev-qt5 \
        libtbb-dev \
        lsb-release \
        make \
        pcl-tools \
        protobuf-compiler \
        python3-tk \
        qtbase5-dev \
        ros-${ROS_VERSION}-desktop \
        ros-${ROS_VERSION}-octomap \
        ros-${ROS_VERSION}-octomap-ros \
        ros-${ROS_VERSION}-octomap-msgs \
        ros-${ROS_VERSION}-octomap-server \
        ros-${ROS_VERSION}-rviz \
        ros-${ROS_VERSION}-vision-msgs \
        sudo \
        tar \
        tmux \
        vim \
        wget \
        xdg-user-dirs \
        xsel \
    && \
    \
    echo "source /opt/ros/${ROS_VERSION}/setup.bash" >> $HOME/.bashrc && \
    echo "source $HOME/simulator/devel/setup.bash" >> $HOME/.bashrc && \
    \
    \
    pip install \
        carla==${CARLA_VERSION} \
        extensisq \
        jax \
        jaxlib \
        jaxtyping \
        opencv-python \
        pandas \
        pygame \
        rosdep \
        rosinstall \
        rosinstall-generator \
        scikit-spatial \
        scipy \
        setuptools \
        typing_extensions \
        torch \
        torchaudio \
        torchvision \
        wstool

COPY jax_guam /jax_guam
WORKDIR /jax_guam
RUN pip install -e . && \
    \
    \
    ln -s /usr/include/opencv4/opencv2 /usr/include/opencv2 && \
    pip install --force netifaces attrs && \
    /usr/local/bin/create_links.sh && rm /usr/local/bin/create_links.sh