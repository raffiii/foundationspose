import json, random
import numpy as np
from simpub.core.net_manager import init_net_manager, logger
from simpub.xr_device.xr_device import XRDevice
from functools import partial

def send_point_cloud(self, point_data) -> None:
    self.request_socket.send_string(json.dumps(self.point_cloud_data))


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
            positions.extend([-float(x / width), float(y / height), -float(depth_pixel)])
            colors.extend([int(rgb_pixel[0]) / 255, int(rgb_pixel[1]) / 255, int(rgb_pixel[2]) / 255, 1])
    return json.dumps({"positions": positions, "colors": colors})

def get_bbox(msg, color = None, depth = None, folder = "debug", file = None):
    o = json.loads(msg)
    center_pose = np.float64(o["center_pose"])
    bbox = np.float64(o["bbox"])
    if file is not None:
        np.savez(f"{folder}/labeled/{file}", color = color, depth = depth, bbox = bbox, center_pose = center_pose)



def send_saved_point_cloud(folder,file):
    path = f"{folder}/{file}"
    data = np.load(path)
    color = data['depth'][...,::-1]
    depth=data['color']
    send_data = generate_point_cloud_data(color,depth)
    net_manager = init_net_manager("192.168.0.134")
    unity_editor = XRDevice("ALRMetaQuest3")
    unity_editor.register_topic_callback("bbox_submission", partial(get_bbox,color = color, depth = depth, folder = folder, file = file))
    # print(generate_point_cloud_data(rgb_image, depth_image))
    while unity_editor.connected is False:
        pass
    unity_editor.request("LoadPointCloud", send_data)
    net_manager.join()

if __name__ == "__main__":

    send_saved_point_cloud("debug/frames","1724146523748.npz")
