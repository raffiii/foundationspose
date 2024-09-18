docker rm -f foundationpose
DIR=$(pwd)/../
xhost +  && docker run \
    --gpus all \
    --env NVIDIA_DISABLE_REQUIRE=1 \
    -it \
    --network=host \
    --name foundationpose  \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    -v $DIR:$DIR \
    -v /home:/home \
    -v /mnt:/mnt \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v /tmp:/tmp  \
    -v /dev:/dev \
    --device-cgroup-rule "c 81:* rmw" \
    --device-cgroup-rule "c 189:* rmw" \
    --ipc=host \
    -e DISPLAY=${DISPLAY} \
    -e GIT_INDEX_FILE \
    -p "5678:5678" \
    foundationspose/realsense:latest bash \
    -c "cd $DIR && bash"
