docker run -it \
-v "$(pwd)/../":/workspace \
-v /tmp/.X11-unix:/tmp/.X11-unix:ro \
-e DISPLAY=unix$DISPLAY \
--net=host \
--gpus all \
--privileged \
--name vision_trt_refactored \
rosvision:xingyou_0312


