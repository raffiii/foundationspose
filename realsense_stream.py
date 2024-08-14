import queue
import threading
from datetime import datetime, timedelta

import cv2
import depthai as dai
import numpy as np
import pyrealsense2 as rs
from stream import CameraStream, DepthFrame, multistream


def realsense_intrinsics(pipeline):
    profile = pipeline.get_active_profile()
    color_stream = profile.get_stream(rs.stream.depth)
    intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
    K = np.array(
        [
            [intrinsics.fx, 0.0, intrinsics.ppx],
            [0.0, intrinsics.fy, intrinsics.ppy],
            [0.0, 0.0, 1.0],
        ]
    )
    return K


# Function to stream from the RealSense camera
def stream_realsense(stream: CameraStream):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    pipeline.start(config)
    try:
        stream.K = realsense_intrinsics(pipeline)
        while stream.running:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue
            # Process the frames (e.g., display or save)
            depth = np.asanyarray(depth_frame.get_data()).astype(np.float16)
            depth = depth / np.max(depth) * 4.0
            color = (
                np.asanyarray(color_frame.get_data())
                .astype(np.float32)[..., ::-1]
                .copy()
            )
            stream.queue.put(DepthFrame(color, depth, datetime.now(), id))
            print(f"Put frames for cam: realsense", end="\r")
    finally:
        pipeline.stop()


# Weights to use when blending depth/rgb image (should equal 1.0)
rgbWeight = 0.4
depthWeight = 0.6


def updateBlendWeights(percent_rgb):
    """
    Update the rgb and depth weights used to blend depth/rgb image

    @param[in] percent_rgb The rgb weight expressed as a percentage (0..100)
    """
    global depthWeight
    global rgbWeight
    rgbWeight = float(percent_rgb) / 100.0
    depthWeight = 1.0 - rgbWeight


def show_images(stream: list):
    # Configure windows; trackbar adjusts blending ratio of rgb/depth
    rgbWindowName = "rgb"
    depthWindowName = "depth"
    blendedWindowName = "rgb-depth"
    cv2.namedWindow(rgbWindowName)
    cv2.namedWindow(depthWindowName)
    cv2.namedWindow(blendedWindowName)
    cv2.createTrackbar(
        "RGB Weight %",
        blendedWindowName,
        int(rgbWeight * 100),
        100,
        updateBlendWeights,
    )
    while True:
        frame: DepthFrame = stream[0].queue.get()
        # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(frame.depth, alpha=0.03), cv2.COLORMAP_JET)
        cv2.imshow(rgbWindowName, frame.color)
        # cv2.imshow(depthWindowName, frame.depth)
        # blended = cv2.addWeighted(frame.color, rgbWeight, depth_colormap, depthWeight, 0)
        # cv2.imshow(blendedWindowName, blended)
        if cv2.waitKey(1) == ord("q"):
            stream[0].stop()
            break

def available_streams():
    return {"realsense": stream_realsense}

if __name__ == "__main__":
    multistream(show_images, available_streams())
