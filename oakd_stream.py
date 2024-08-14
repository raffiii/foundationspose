#!/usr/bin/env python3
from datetime import datetime, timedelta

import cv2
import depthai as dai
import numpy as np
from stream import CameraStream, DepthFrame, multistream

# Weights to use when blending depth/rgb image (should equal 1.0)
rgbWeight = 0.4
depthWeight = 0.6


def oakd_intrinsics(device):
    calibData = device.readCalibration()
    intrinsics = calibData.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A)
    return intrinsics


def updateBlendWeights(percent_rgb):
    """
    Update the rgb and depth weights used to blend depth/rgb image

    @param[in] percent_rgb The rgb weight expressed as a percentage (0..100)
    """
    global depthWeight
    global rgbWeight
    rgbWeight = float(percent_rgb) / 100.0
    depthWeight = 1.0 - rgbWeight


def run_oakd(
    stream: CameraStream,
    fps=30,
    monoResolution=dai.MonoCameraProperties.SensorResolution.THE_480_P,
    rgbResolution=dai.ColorCameraProperties.SensorResolution.THE_1080_P,
    rgbCamSocket=dai.CameraBoardSocket.CAM_A,
):
    # Create pipeline
    pipeline = dai.Pipeline()
    queueNames = []

    # Define sources and outputs
    camRgb = pipeline.create(dai.node.ColorCamera)
    left = pipeline.create(dai.node.MonoCamera)
    right = pipeline.create(dai.node.MonoCamera)
    stereo = pipeline.create(dai.node.StereoDepth)
    sync = pipeline.create(dai.node.Sync)
    demux = pipeline.create(dai.node.MessageDemux)

    rgbOut = pipeline.create(dai.node.XLinkOut)
    disparityOut = pipeline.create(dai.node.XLinkOut)

    rgbOut.setStreamName("rgb")
    queueNames.append("rgb")
    disparityOut.setStreamName("disp")
    queueNames.append("disp")

    # Properties
    camRgb.setBoardSocket(rgbCamSocket)
    camRgb.setResolution(rgbResolution)
    camRgb.setFps(fps)

    left.setResolution(monoResolution)
    left.setCamera("left")
    left.setFps(fps)
    right.setResolution(monoResolution)
    right.setCamera("right")
    right.setFps(fps)

    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    # LR-check is required for depth alignment
    stereo.setLeftRightCheck(True)
    stereo.setDepthAlign(rgbCamSocket)

    # Set sync threshold
    sync.setSyncThreshold(timedelta(milliseconds=100))

    # Linking
    left.out.link(stereo.left)
    right.out.link(stereo.right)
    camRgb.video.link(sync.inputs["rgb"])
    stereo.disparity.link(sync.inputs["depth"])
    sync.out.link(demux.input)
    demux.outputs["rgb"].link(rgbOut.input)
    demux.outputs["depth"].link(disparityOut.input)

    # Connect to device and start pipeline
    print(f"Connecting to OAK-D device {stream.cam_id}")
    device_info = dai.DeviceInfo(stream.cam_id)
    with dai.Device(pipeline, deviceInfo=device_info) as device:
        stream.K = oakd_intrinsics(device)
        # For now, RGB needs fixed focus to properly align with depth.
        # This value was used during calibration
        try:
            calibData = device.readCalibration2()
            lensPosition = calibData.getLensPosition(rgbCamSocket)
            if lensPosition:
                camRgb.initialControl.setManualFocus(lensPosition)
        except:
            raise

        frameRgb = None
        frameDisp = None

        while stream.running:
            latestPacket = {}
            latestPacket["rgb"] = None
            latestPacket["disp"] = None

            queueEvents = device.getQueueEvents(("rgb", "disp"))
            for queueName in queueEvents:
                packets = device.getOutputQueue(queueName).tryGetAll()
                if len(packets) > 0:
                    latestPacket[queueName] = packets[-1]

            if latestPacket["rgb"] is not None:
                frameRgb = latestPacket["rgb"].getCvFrame()
            if latestPacket["disp"] is not None:
                frameDisp = latestPacket["disp"].getFrame()
                maxDisparity = stereo.initialConfig.getMaxDisparity()
                if 1:
                    frameDisp = (frameDisp * 255.0 / maxDisparity).astype(np.uint8)
                # Optional, apply false colorization
                if 1:
                    frameDisp = cv2.applyColorMap(frameDisp, cv2.COLORMAP_HOT)
                frameDisp = np.ascontiguousarray(frameDisp)
            if frameRgb is not None and frameDisp is not None:
                frame = DepthFrame(frameRgb, frameDisp, datetime.now(), stream.cam_id)
                stream.queue.put(frame)
                print(f"Put frame in queue at {frame.time}", end="\r")
                frameRgb = None
                frameDisp = None
            else:
                print(f"Not both frames at {datetime.now()}", end="\r")


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
        cv2.imshow(rgbWindowName, frame.color)
        cv2.imshow(depthWindowName, frame.depth)
        if len(frame.depth.shape) < 3:
            frame.depth = cv2.cvtColor(frame.depth, cv2.COLOR_GRAY2BGR)
        blended = cv2.addWeighted(frame.color, rgbWeight, frame.depth, depthWeight, 0)
        cv2.imshow(blendedWindowName, blended)
        if cv2.waitKey(1) == ord("q"):
            stream[0].stop()
            break


def available_streams(num=-1):
    return {
        device.getMxId(): run_oakd
        for i, device in enumerate(dai.Device.getAllAvailableDevices())
        if num < 0 or i < num
    }


if __name__ == "__main__":
    multistream(show_images, available_streams(1))
    # def stopper():
    #     stream.running = False
    # stream = CameraStream("18443010A1A7701200",stopper)
    # cam = threading.Thread(target=run_oakd, args=(stream,))
    # view = threading.Thread(target=show_images, args=([stream],))
    # cam.start()
    # view.start()
    # view.join()
    # cam.join()
