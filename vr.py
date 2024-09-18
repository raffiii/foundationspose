import json, random
import numpy as np
from simpub.core.net_manager import init_net_manager, logger
from simpub.xr_device.xr_device import XRDevice
from functools import partial
import argparse
import os, os.path
import threading, time
from scipy.spatial.transform import Rotation
import cv2


def generate_point_cloud_data(rgb_image, depth_image):
    positions = []
    colors = []
    height, width, _ = rgb_image.shape
    for y in range(height):
        for x in range(width):
            rgb_pixel = rgb_image[y, x]
            depth_pixel = depth_image[y, x]
            if random.random() > 0.1:
                continue
            positions.extend([float(x / width), -float(y / height), float(depth_pixel)])
            colors.extend(
                [
                    int(rgb_pixel[0]) / 255,
                    int(rgb_pixel[1]) / 255,
                    int(rgb_pixel[2]) / 255,
                    1,
                ]
            )
    return json.dumps({"positions": positions, "colors": colors})


def get_bbox(msg, use_bbox, stop = lambda: None):
    o = json.loads(msg)
    logger.info(f"Got bbox msg: {o}")
    data = o["data"][0]
    pos = np.float64(data[:3])
    rot = np.float64(data[3:7])
    sca = np.float64(data[7:])
    center_pose, bbox = from_quaternion(pos, rot, sca)
    use_bbox(bbox=bbox, center_pose=center_pose)
    stop()


def bbox2msg(bbox, center_pose):
    pos, rot, sca = to_quaternion(center_pose, bbox)
    data = [i.item() for l in [pos, rot, sca] for i in l]
    o = json.dumps({"data": [data]})
    return o


def save_bbox(color, depth, bbox, center_pose, folder, file):
    if file is not None:
        from pathlib import Path

        path = f"{folder}/labeled"
        Path(path).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created path {path}")
        np.savez(
            f"{path}/{file}",
            color=color,
            depth=depth,
            bbox=bbox,
            center_pose=center_pose,
        )
    


def from_quaternion(translation, quaternion, local_scale):
    S = np.diag(local_scale)
    T = np.eye(4)
    T[:3, 3] = translation
    R = Rotation.from_quat(quaternion).as_matrix()
    T[:3, :3] = R
    T[:3, :3] *= S
    S = np.float64(local_scale)
    S /= 2
    return T, np.float64([S, -S])


def to_quaternion(T, scaled_values):
    # Extract the translation
    translation = T[:3, 3]

    # Extract the scale
    S = scaled_values[0] * 2

    # Extract the rotation matrix and normalize it by the scale
    R = T[:3, :3] / S

    # Convert the rotation matrix back to a quaternion
    quaternion = Rotation.from_matrix(R).as_quat()

    return translation, quaternion, S


def to_quaternion(T, S):
    T, S = np.float64(T), np.float64(S)
    s = S[0] - S[1]
    t = T[:3, 3]
    r = T[:3, :3]
    local_scale = np.linalg.norm(r)
    r = r / local_scale
    q = Rotation.from_matrix(r).as_quat()

def waiting():
    is_running = []
    def run():
        while len(is_running) == 0:
            time.sleep(0.1)
    def stop(): 
        is_running.append(0) 
    wait = threading.Thread(target = run)
    return wait, stop
    

def send_saved_point_cloud(folder, files, ip):
    # net_manager = init_net_manager("10.10.10.220")
    net_manager = init_net_manager(ip)
    net_manager.start()
    unity_editor = XRDevice("ALRMetaQuest3")
    for file in files:
        path = f"{folder}/{file}"
        data = np.load(path)
        color = data["color"]
        depth = data["depth"]
        use_bbox = partial(save_bbox, color=color, depth=depth, folder=folder, file=file)
        await_label, stop = waiting()
        unity_editor.register_topic_callback(
            "bbox_submission",
            partial(get_bbox, use_bbox=use_bbox, stop = stop)
            
        )
        send_point_cloud(net_manager, unity_editor, color, depth)
        cv2.imshow("color", color)
        await_label.start()
        await_label.join()


def send_bbox(bbox, center_pose, unity_editor):
    s = bbox2msg(bbox, center_pose)
    while unity_editor.connected is False:
        pass
    unity_editor.request("LoadBBox", s)


def send_point_cloud(net_manager, unity_editor, color, depth):

    send_data = generate_point_cloud_data(color, depth)
    # print(generate_point_cloud_data(rgb_image, depth_image))
    while unity_editor.connected is False:
        pass
    unity_editor.request("LoadPointCloud", send_data)


def send_live_point_cloud(ip, color, depth, use_bbox):
    net_manager = init_net_manager(ip)
    net_manager.start()
    unity_editor = XRDevice("ALRMetaQuest3")
    send_point_cloud(net_manager, unity_editor, color[..., ::-1], depth)
    unity_editor.register_topic_callback(
        "bbox_submission",
        partial(get_bbox, use_bbox=use_bbox),
    )
    return net_manager


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--file", default=None)
    parser.add_argument("--ip", default="192.168.0.134")
    opts = parser.parse_args()

    # check if whole directory is to label
    if opts.file is None:
        files = [f for f in os.listdir(opts.path) if os.path.isfile(os.path.join(opts.path, f))]
    else:
        files = [opts.file]


    # data_path = "debug/frames"
    # file_name = "1724146523748c.npz"
    # ip_addr = "192.168.0.103"

    # send_saved_point_cloud(data_path, file_name, ip_addr)
    send_saved_point_cloud(opts.path, files, opts.ip)
    # send_saved_point_cloud("debug/frames","1724146523748.npz")
