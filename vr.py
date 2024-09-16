import json, random
import numpy as np
from SimPublisher.simpub.core.net_manager import init_net_manager, logger
from SimPublisher.simpub.xr_device.xr_device import XRDevice
from functools import partial
import argparse
from scipy.spatial.transform import Rotation


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


def get_bbox(msg, use_bbox):
    o = json.loads(msg)
    logger.info(f"Got bbox msg: {o}")
    data = o["data"][0]
    pos = np.float64(data[:3])
    rot = np.float64(data[3:7])
    sca = np.float64(data[7:])
    center_pose, bbox = from_quaternion(pos, rot, sca)
    use_bbox(bbox=bbox, center_pose=center_pose)

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


def to_quaternion(T, S):
    T, S = np.float64(T), np.float64(S)
    s = S[0] - S[1]
    t = T[:3, 3]
    r = T[:3, :3]
    local_scale = np.linalg.norm(r)
    r = r / local_scale
    q = Rotation.from_matrix(r).as_quat()


def send_saved_point_cloud(folder, file, ip):
    path = f"{folder}/{file}"
    data = np.load(path)
    color = data["color"]
    depth = data["depth"]
    #net_manager = init_net_manager("10.10.10.220")
    net_manager = init_net_manager(ip)
    net_manager.start()
    unity_editor = XRDevice("ALRMetaQuest3")
    use_bbox = partial(save_bbox, color=color, depth=depth, folder=folder, file=file)
    unity_editor.register_topic_callback(
        "bbox_submission",
        partial(get_bbox, use_bbox=use_bbox),
    )
    send_point_cloud(net_manager, unity_editor, color, depth)
    net_manager.join()


def send_point_cloud(net_manager, unity_editor, color, depth):
    send_data = generate_point_cloud_data(color, depth)
    # print(generate_point_cloud_data(rgb_image, depth_image))
    while unity_editor.connected is False:
        pass
    unity_editor.request("LoadPointCloud", send_data)

def send_live_point_cloud(ip,color, depth, use_bbox):
    net_manager = init_net_manager(ip)
    unity_editor = XRDevice("ALRMetaQuest3")
    send_point_cloud(net_manager, unity_editor, color, depth)
    unity_editor.register_topic_callback(
        "bbox_submission",
        partial(get_bbox, use_bbox=use_bbox),
    )
    return net_manager
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("file")
    parser.add_argument('--ip',default="192.168.0.134")
    opts = parser.parse_args()

    # data_path = "debug/frames"
    # file_name = "1724146523748c.npz"
    # ip_addr = "192.168.0.103"

    # send_saved_point_cloud(data_path, file_name, ip_addr)
    send_saved_point_cloud(opts.path, opts.file, opts.ip)
    # send_saved_point_cloud("debug/frames","1724146523748.npz")
