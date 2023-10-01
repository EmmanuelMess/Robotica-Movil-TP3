FROM ros:humble-perception

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    lsb-release \
    wget \
    gnupg \
    python3-pip \
    ros-humble-turtlesim \
    ros-humble-rqt* \
    ros-humble-navigation2 \
    ros-humble-nav2-bringup \
    ros-humble-turtlebot3* \
    ros-humble-gazebo-* \
    ros-humble-laser-geometry \
    ros-humble-sensor-msgs \
    ros-humble-demo-nodes-cpp \
    ros-humble-demo-nodes-py && \
    rm -rf /var/lib/apt/lists/*

ADD ./requirements.txt /tmp/requirements.txt

RUN python3 -m pip install -r /tmp/requirements.txt

RUN ["/bin/bash", "-c", "source /opt/ros/humble/setup.bash"]
