import depthai as dai
import pyrealsense2 as rs
import threading
import queue
import argparse
import numpy as np

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
    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    camRgb.setVideoSize(1920, 1080)

    xoutVideo.input.setBlocking(False)
    xoutVideo.input.setQueueSize(1)

    # Linking
    camRgb.video.link(xoutVideo.input)

    device_info = dai.DeviceInfo(device_id)
    # with dai.Device(pipeline, device_info) as device:

    with dai.Device(pipeline,deviceInfo=device_info) as device:
        intrinsics.put(oakd_intrinsics(device))
        # Output queue will be used to get the disparity frames from the outputs defined above
        q = device.getOutputQueue(name=f"disparity_{device_id}", maxSize=4, blocking=False)
        q_rgb = device.getOutputQueue(name=f"rgb_{device_id}", maxSize=4, blocking=False)

        while stopper.empty():
            inDisparity = q.get()  # blocking call, will wait until a new data has arrived
            frame = inDisparity.getFrame()
            # Normalization for better visualization
            frame = (frame * (255 / depth.initialConfig.getMaxDisparity())).astype(np.uint8)

            in_rgb = q_rgb.get()
            rgb = in_rgb.getCvFrame()
            frame_q.put((rgb,frame))
            print(f"Put frames for cam: {device_id}", end="\r")
            

def realsense_intrinsics(pipeline):
    profile = pipeline.get_active_profile()
    color_stream = profile.get_stream(rs.stream.depth)
    intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
    K = np.array([[intrinsics.fx,0.0,intrinsics.ppx],[0.0,intrinsics.fy, intrinsics.ppy],[0.0,0.0,1.0]])
    return K

# Function to stream from the RealSense camera
def stream_realsense(frame_q, stopper,intrinsics):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    pipeline.start(config)
    try:
        intrinsics.put(realsense_intrinsics(pipeline))
        while stopper.empty():
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue
            # Process the frames (e.g., display or save)
            depth = np.asanyarray(depth_frame.get_data()).astype(np.float16)
            depth = depth / np.max(depth) * 4.0
            color = np.asanyarray(color_frame.get_data()).astype(np.float32)[...,::-1].copy()
            frame_q.put((color,depth))
            print(f"Put frames for cam: realsense", end="\r")
    finally:
        pipeline.stop()


def multistream(f=None, rslive=True,oakd=True):
    # create a queue to insert all queues in
    streams = []
    stopper = queue.Queue()
    
    threads = []

    if oakd:
        # Get OAK-D Camera ids
        if oakd is True:
            oakd_device_ids = [device.getMxId() for device in dai.Device.getAllAvailableDevices()]
            print(oakd_device_ids)
        else:
            oakd_device_ids=oakd
        
        # Create threads for each camera stream
        for device_id in oakd_device_ids:
            q = queue.Queue()
            intrinsics = queue.Queue()
            t = threading.Thread(target=stream_oakd, args=(device_id,q, stopper, intrinsics))
            threads.append(t)
            streams.append({"frames":q,"intrinsics":intrinsics})
    if rslive:
        q = queue.Queue()
        intrinsics = queue.Queue()
        realsense_thread = threading.Thread(target=stream_realsense, args=(q,stopper, intrinsics))
        threads.append(realsense_thread)
        streams.append({"frames":q,"intrinsics":intrinsics})

    # Call the function consuming the frames
    if f:
        threads.append(threading.Thread(target=f, args=(streams,)))

    # Start the threads
    for t in threads:
        t.start()

    input("")
    stopper.put(1)

    # Join the threads
    for t in threads:
        t.join()


if __name__ == '__main__':
    multistream()