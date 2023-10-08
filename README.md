## Setup
### 1) En una terminal levantamos el sistema
```bash
bash dev/setup.sh
docker compose build
docker compose up -d
```

### 2) Levantamos la simulacion
```bash
docker exec -it simulation_env bash
source /opt/ros/humble/setup.bash && . /usr/share/gazebo/setup.bash && ros2 launch stereo_image_proc stereo_image_proc.launch.py
```

### 3) Levantar el rosbag
```bash
docker exec -it simulation_env bash
source /opt/ros/humble/setup.bash && . /usr/share/gazebo/setup.bash && ros2 bag play /catkin_ws/src/rosbag2_2022_11_09-15_21_22/  --loop --remap /stereo/left/image_raw:=/left/image_raw /stereo/left/camera_info:=/left/camera_info /stereo/right/image_raw:=/right/image_raw /stereo/right/camera_info:=/right/camera_info
```

### 4) Levantar el nodo nuestro
```bash
docker exec -it simulation_env bash
source /opt/ros/humble/setup.bash && . /usr/share/gazebo/setup.bash && python3 /catkin_ws/src/image_server.py
```