import depthai as dai
import cv2
import numpy as np

# Create pipeline
pipeline = dai.Pipeline()

# Define sources and outputs
cam_rgb = pipeline.create(dai.node.ColorCamera)
stereo = pipeline.create(dai.node.StereoDepth)
xout_rgb = pipeline.create(dai.node.XLinkOut)
xout_depth = pipeline.create(dai.node.XLinkOut)

xout_rgb.setStreamName("rgb")
xout_depth.setStreamName("depth")

# Properties
cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam_rgb.setInterleaved(False)
cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

# StereoDepth
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
stereo.setOutputDepth(True)
stereo.setOutputRectified(True)
stereo.setConfidenceThreshold(200)

# Linking
cam_rgb.video.link(xout_rgb.input)
stereo.depth.link(xout_depth.input)

# Connect to device and start pipeline
with dai.Device(pipeline) as device:
    # Output queues
    q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    q_depth = device.getOutputQueue(name="depth", maxSize=4, blocking=False)

    while True:
        in_rgb = q_rgb.get()
        in_depth = q_depth.get()

        # Get BGR frame
        frame_rgb = in_rgb.getCvFrame()

        # Get depth frame
        frame_depth = in_depth.getFrame()
        frame_depth = (frame_depth * (255 / stereo.getMaxDisparity())).astype(np.uint8)
        frame_depth = cv2.applyColorMap(frame_depth, cv2.COLORMAP_JET)

        # Display frames
        cv2.imshow("rgb", frame_rgb)
        cv2.imshow("depth", frame_depth)

        if cv2.waitKey(1) == ord('q'):
            break
