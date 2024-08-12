import queue
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta

import cv2
import depthai as dai
import numpy as np
import pyrealsense2 as rs


@dataclass
class DepthFrame:
    color: np.array
    depth: np.array
    time: datetime
    id: str


def oakd_intrinsics(device):
    calibData = device.readCalibration()
    intrinsics = calibData.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A)
    return intrinsics


# Function to stream from an OAK-D camera
def stream_oakd(device_id, frame_q, stopper, intrinsics):
    # Closer-in minimum depth, disparity range is doubled (from 95 to 190):
    extended_disparity = False
    # Better accuracy for longer distance, fractional disparity 32-levels:
    subpixel = False
    # Better handling for occlusions:
    lr_check = True

    # Create pipeline
    pipeline = dai.Pipeline()

    # Define sources and outputs
    monoLeft = pipeline.create(dai.node.MonoCamera)
    monoRight = pipeline.create(dai.node.MonoCamera)
    depth = pipeline.create(dai.node.StereoDepth)
    xout = pipeline.create(dai.node.XLinkOut)

    xout.setStreamName(f"disparity_{device_id}")

    # Properties
    monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoLeft.setCamera("left")
    monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoRight.setCamera("right")

    # Create a node that will produce the depth map (using disparity output as it's easier to visualize depth this way)
    depth.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    # Options: MEDIAN_OFF, KERNEL_3x3, KERNEL_5x5, KERNEL_7x7 (default)
    depth.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
    depth.setLeftRightCheck(lr_check)
    depth.setExtendedDisparity(extended_disparity)
    depth.setSubpixel(subpixel)

    # Linking
    monoLeft.out.link(depth.left)
    monoRight.out.link(depth.right)
    depth.disparity.link(xout.input)

    # Define source and output
    camRgb = pipeline.create(dai.node.ColorCamera)
    xoutVideo = pipeline.create(dai.node.XLinkOut)

    xoutVideo.setStreamName(f"rgb_{device_id}")

    # Properties
    camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_)
    camRgb.setVideoSize(600, 400)

    xoutVideo.input.setBlocking(False)
    xoutVideo.input.setQueueSize(1)

    # Linking
    camRgb.video.link(xoutVideo.input)

    device_info = dai.DeviceInfo(device_id)
    # with dai.Device(pipeline, device_info) as device:

    with dai.Device(pipeline, deviceInfo=device_info) as device:
        intrinsics[id] = oakd_intrinsics(device)
        # Output queue will be used to get the disparity frames from the outputs defined above
        q = device.getOutputQueue(
            name=f"disparity_{device_id}", maxSize=4, blocking=False
        )
        q_rgb = device.getOutputQueue(
            name=f"rgb_{device_id}", maxSize=4, blocking=False
        )

        while stopper.empty():
            inDisparity = (
                q.get()
            )  # blocking call, will wait until a new data has arrived
            frame = inDisparity.getFrame()
            # Normalization for better visualization
            frame = (frame * (255 / depth.initialConfig.getMaxDisparity())).astype(
                np.uint8
            )

            in_rgb = q_rgb.get()
            rgb = in_rgb.getCvFrame()
            frame_q.put(DepthFrame(rgb, frame, datetime.now(), id))
            print(f"Put frames for cam: {device_id}", end="\r")


def stream_oakd_stereo(device_id, frame_q, stopper, intrinsics):
    FPS = 30.0

    RGB_SOCKET = dai.CameraBoardSocket.CAM_A
    LEFT_SOCKET = dai.CameraBoardSocket.CAM_B
    RIGHT_SOCKET = dai.CameraBoardSocket.CAM_C
    ALIGN_SOCKET = LEFT_SOCKET

    COLOR_RESOLUTION = dai.ColorCameraProperties.SensorResolution.THE_1080_P
    LEFT_RIGHT_RESOLUTION = dai.MonoCameraProperties.SensorResolution.THE_400_P

    ISP_SCALE = 3

    pipeline = dai.Pipeline()

    # Define sources and outputs
    camRgb = pipeline.create(dai.node.ColorCamera)
    left = pipeline.create(dai.node.MonoCamera)
    right = pipeline.create(dai.node.MonoCamera)
    stereo = pipeline.create(dai.node.StereoDepth)
    sync = pipeline.create(dai.node.Sync)
    out = pipeline.create(dai.node.XLinkOut)
    align = pipeline.create(dai.node.ImageAlign)

    left.setResolution(LEFT_RIGHT_RESOLUTION)
    left.setBoardSocket(LEFT_SOCKET)
    left.setFps(FPS)

    right.setResolution(LEFT_RIGHT_RESOLUTION)
    right.setBoardSocket(RIGHT_SOCKET)
    right.setFps(FPS)

    camRgb.setBoardSocket(RGB_SOCKET)
    camRgb.setResolution(COLOR_RESOLUTION)
    camRgb.setFps(FPS)
    camRgb.setIspScale(1, ISP_SCALE)

    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    stereo.setDepthAlign(dai.CameraBoardSocket.LEFT)

    out.setStreamName(f"out_{device_id}")

    sync.setSyncThreshold(timedelta(seconds=(1 / FPS) * 0.5))

    # Linking
    camRgb.isp.link(sync.inputs["rgb"])
    left.out.link(stereo.left)
    right.out.link(stereo.right)
    stereo.depth.link(align.input)
    align.outputAligned.link(sync.inputs["depth_aligned"])
    camRgb.isp.link(align.inputAlignTo)
    sync.out.link(out.input)

    device_info = dai.DeviceInfo(device_id)

    with dai.Device(pipeline, deviceInfo=device_info) as device:
        calibrationHandler = device.readCalibration()
        rgbIntrinsics = calibrationHandler.getCameraIntrinsics(
            RGB_SOCKET, int(1920 / ISP_SCALE), int(1080 / ISP_SCALE)
        )
        rgbDistortion = calibrationHandler.getDistortionCoefficients(RGB_SOCKET)
        intrinsics[id] = rgbIntrinsics
        q = device.getOutputQueue(name=f"out_{device_id}", maxSize=4, blocking=False)

        while stopper.empty():
            messageGroup = q.get()
            frameRgb = messageGroup["rgb"]
            frameDepth = messageGroup["depth_aligned"]

            # Undistort

            if frameDepth is not None and frameRgb is not None:
                cvFrame = frameDepth.getCvFrame()
                cvFrameUndistorted = cv2.undistort(
                    cvFrame,
                    np.array(rgbIntrinsics),
                    np.array(rgbDistortion),
                )
                depth = frameDepth.getFrame()
                vis = np.hstack((cvFrame,cvFrameUndistorted))
                #vis = depth
                #cv2.imshow('1', vis[...,::1])
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                frame_q.put(
                    DepthFrame(
                        color=cvFrameUndistorted[..., ::-1],
                        depth=depth,
                        time=datetime.now(),
                        id=id,
                    )
                )
                print(f"Put frames for cam: {device_id} with depth dimension {depth.shape} and color dimension {cvFrameUndistorted.shape}", end="\r")
        print("")
        print(f"Stopped thread for camera {device_id}")

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
def stream_realsense(frame_q, stopper, intrinsics):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    pipeline.start(config)
    try:
        intrinsics["realsense"] = realsense_intrinsics(pipeline)
        while stopper.empty():
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
            frame_q.put(DepthFrame(color, depth, datetime.now(),id))
            print(f"Put frames for cam: realsense", end="\r")
    finally:
        pipeline.stop()


def multistream(f=None, rslive=True, oakd=True):
    # create a queue to insert all queues in
    q = queue.Queue(1000)
    stopper = queue.Queue()
    intrinsics = {}
    ids = []
    threads = []

    if oakd:
        # Get OAK-D Camera ids
        if oakd is True or type(oakd) is int:
            oakd_device_ids = [
                device.getMxId() for device in dai.Device.getAllAvailableDevices()
            ]
            print(oakd_device_ids)
            if type(oakd) is int:
                oakd_device_ids = oakd_device_ids[:oakd]
        else:
            oakd_device_ids = oakd
        ids += oakd_device_ids

        # Create threads for each camera stream
        for device_id in oakd_device_ids:
            t = threading.Thread(
                target=stream_oakd_stereo, args=(device_id, q, stopper, intrinsics)
            )
            threads.append(t)
    if rslive:
        realsense_thread = threading.Thread(
            target=stream_realsense, args=(q, stopper, intrinsics)
        )
        threads.append(realsense_thread)
        ids += ["realsense"]

    # Call the function consuming the frames
    if f:
        threads.append(threading.Thread(target=f, args=(q, intrinsics, ids, stopper)))

    # Start the threads
    for t in threads:
        t.start()
    print(f"Started {len(threads)} threads.\n")
    input("")
    stopper.put(1)

    # Join the threads
    for t in threads:
        t.join()


if __name__ == "__main__":
    def f(q,*args):
        while True:
            q.get()
    multistream(rslive=False, oakd=1, f=f)
