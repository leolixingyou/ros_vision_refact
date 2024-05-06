# hpp-pointcloud-realtime


## How to use
1. ./docker_build.sh
2. ./run_container.sh
3. (in container) ./run_main.sh
4. Ctrl+C and $exit to exit


## How to set ROS network
1. $ifconfig => check your ip (192.168.0.XXX)
2. vi workspace/run_main.sh
3. replace ROS_IP=192.168.0.XXX with your ip
4. do the same for viz.sh