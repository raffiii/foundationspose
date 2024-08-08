import open3d as o3d
import numpy as np

def to_point_cloud(rgb, depth, T):
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        rgb_image, 
        depth_image, 
        convert_rgb_to_intensity=False
    )
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image, 
        T
    )
    # Flip the point cloud for a better view
    pcd.transform([[1, 0, 0, 0],
                   [0, -1, 0, 0],
                   [0, 0, -1, 0],
                   [0, 0, 0, 1]])
    return pcd

def save_ply(rgb, depth, T, path="point_cloud.ply"):
    pcd = to_point_cloud(rgb, depth, T)
    # Save the point cloud to a file
    o3d.io.write_point_cloud(path, pcd)

