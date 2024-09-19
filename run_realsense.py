import pyrealsense2 as rs
import numpy as np
import cv2
import trimesh
import argparse
import time
import logging

from estimater import *


def run_pose_estimation_worker(
    color,
    depth,
    K,
    est: FoundationPose,
    opts,
    ob_mask=None,
    debug=0,
    ob_id=None,
    device="cuda:0",
):
    if ob_mask is not None:
        logging.info("Registering object")
        pose = est.register(K=K, rgb=color, depth=depth, ob_mask=ob_mask)
        logging.info(f"pose:\n{pose}")
        return pose
    else:
        logging.info("Tracking registered object")
        try:
            pose = est.track_one(
                rgb=color, depth=depth, K=K, iteration=opts.track_refine_iter
            )
            return pose
        except:
            logging.warn("pose tracking failed")
        return None


def get_reconstructed_mesh(ob_id, ref_view_dir):
    mesh = trimesh.load(
        os.path.abspath(f"{ref_view_dir}/ob_{ob_id:07d}/model/model.obj")
    )
    return mesh


def build_estimator(mesh, opts):

    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    est = FoundationPose(
        model_pts=mesh.vertices,
        model_normals=mesh.vertex_normals,
        mesh=mesh,
        scorer=scorer,
        refiner=refiner,
        debug_dir=opts.debug_dir,
        debug=opts.debug,
        glctx=glctx,
    )
    logging.info("estimator initialization done")
    return (
        est,
        to_origin,
        bbox,
    )


def reset_cameras():
    """
    Performs a hardware reset on realsense cameras
    Try if cameras don't provide images until a timeout
    """
    print("reset start")
    ctx = rs.context()
    devices = ctx.query_devices()
    for dev in devices:
        dev.hardware_reset()
    print("reset done")


def run_with_pipeline(function, opts):
    """
    Runs the given `function` with a newly created realsense pipeline
    providing depth and color images and further options in `opts`
    """
    reset_cameras()

    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()

    # Start streaming from the default RealSense camera
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_record_to_file(f"{opts.debug_dir}/{opts.save_to}.bag")

    profile = pipeline.start(config)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    result = None
    try:
        result = function(pipeline, depth_scale=depth_scale)
    finally:
        # Stop streaming
        pipeline.stop()
        cv2.destroyAllWindows()
    return result


def run_demo(pipeline, **kwargs):
    """
    Only run realsense camera, included for debugging/test purposes
    """
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET
        )

        # Stack both images horizontally
        images = np.hstack((color_image, depth_colormap))

        # Show images
        cv2.imshow("RealSense", images)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


def run_with_bag(function, path):
    """
    Runs the function with a stored stream as a pipeline rather than a live feed
    """
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()

    # Start streaming from the default RealSense camera
    config.enable_device_from_file("../object_detection.bag")

    pipeline.start(config)
    result = None
    try:
        result = function(pipeline)
    finally:
        # Stop streaming
        pipeline.stop()
        cv2.destroyAllWindows()
    return result


def compute_mask(object, background, threshold=0.05):
    """
    A heuristic to predict the object mask by using the difference between two
    images, without and with the object
    """
    alpha = 1 + threshold
    less, more = (background > alpha * object, object < alpha * background)
    diff = (np.max(more, axis=2) * 1.0 + np.max(less, axis=2) * 1.0) > 0
    return diff


def capture_mask(pipeline):
    """
    Capture the two images for the mask heuristic
    """
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    background = np.asanyarray(color_frame.get_data())
    input("Place object to track and hit enter:")
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    object_image = np.asanyarray(color_frame.get_data())
    return compute_mask(object_image, background)


def fix_mask(mask, maskmask):
    """
    Fix the mask for noise in known background regions
    """
    return (mask * 1.0) * (maskmask * 1.0)


def show_mask():
    """
    Show a captured mask in cv2 imshow
    """
    result = run_with_pipeline(capture_mask)
    shape = result.shape
    result = fix_mask(
        result,
        np.vstack(
            (
                np.zeros((shape[0] // 2, shape[1])),
                np.ones((shape[0] - shape[0] // 2, shape[1])),
            )
        ),
    )
    cv2.imshow("RealSense", result * 1.0)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def mask(pipeline, opts):
    """
    Capture and fix a mask using a heuristic
    """
    mask = capture_mask(pipeline)
    shape = mask.shape
    mask = fix_mask(
        mask,
        np.vstack(
            (
                np.zeros((shape[0] // 2, shape[1])),
                np.ones((shape[0] - shape[0] // 2, shape[1])),
            )
        ),
    )
    # np.save()
    return mask


def parse_start_pose_path(path, est, mat):
    """
    Parse the start pose from a known file
    """
    if path is None:
        return None
    with open(path, "r") as f:
        return parse_start_pose(f, est, mat)


def parse_start_pose(pose, est, mat):
    """
    Parse a pose to use as start pose
    """
    pose = np.load(pose)
    return (pose @ mat) @ est.get_tf_to_centered_mesh()


def save_frame(color, depth, center_pose, bbox, opts, sub_dir="frames"):
    logging.info("saving frame")
    from pathlib import Path

    path = f"{opts.debug_dir}/{sub_dir}/{opts.ob_name}"
    Path(path).mkdir(parents=True, exist_ok=True)
    logging.info(f"Created path {path}")
    np.savez(
        f"{path}/{round(time.time() * 1000)}.npz",
        color=color[..., ::-1],
        depth=depth,
        center_pose=center_pose,
        bbox=bbox,
    )
    logging.info("Frame saved!")


def send_vr(color, depth, opts, bbox = None, center_pose = None):
    def save(bbox, center_pose):
        save_frame(color, depth, center_pose, bbox, opts, sub_dir="frames/labeled")
        

    vr.send_live_point_cloud(opts.simpublish_ip, color, depth, save, bbox=bbox, center_pose=center_pose)


# Precompute the model with run_ycbv/run_linemode in run_nerf.py
def run_live_estimation(opts, get_mask=mask, device="cuda:0", saveFrame=None):
    """
    Create a method to run the estimator with a pipeline, either live or a recorded stream
    """
    mesh = get_reconstructed_mesh(opts.ob_id, opts.ref_view_dir)
    est, to_origin, bbox = build_estimator(mesh, opts)
    if opts.start_pose_path:
        # set known start pose
        pose = parse_start_pose(opts.start_pose_path, est, opts.qrcode_translation)
        est.pose_last = pose

    def run(pipeline, depth_scale=4.0):
        mask = get_mask(pipeline, opts)

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
        np.savez("debug/K.npz", K=K)
        # mask = np.ones((intrinsics.height, intrinsics.width))

        #
        torch.cuda.set_device(device)
        est.to_device(device)
        est.glctx = dr.RasterizeCudaContext(device=device)
        poses = []
        rgbWeight = 0.75

        align_to = rs.stream.color
        align = rs.align(align_to)

        while True:
            # Getting a frame from the realsense camera
            raw_frames = pipeline.wait_for_frames()

            frames = align.process(raw_frames)
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            # Convert images to numpy arrays and apply necessary transformations
            raw_depth = np.asanyarray(depth_frame.get_data())
            # depth = raw_depth / np.max(raw_depth) * 4.0
            depth = raw_depth * depth_scale

            color = np.uint8(color_frame.get_data())

            # run the pose estimation
            pose = run_pose_estimation_worker(
                color[..., ::-1].copy(), depth, K, est, opts, ob_mask=mask
            )
            mask = None
            if pose is not None:
                center_pose = pose @ np.linalg.inv(to_origin)
                poses += [center_pose]
                logging.info(f"Center pose: \n {center_pose}")
            else:
                center_pose = None

            cv2.imshow("color", color)
            vis_base = color
            if opts.show_depth:
                if len(depth.shape) < 3:
                    show_depth = cv2.applyColorMap(
                        cv2.convertScaleAbs(raw_depth, alpha=0.1, beta=10),
                        cv2.COLORMAP_JET,
                    )
                blended = cv2.addWeighted(
                    color, 1 - rgbWeight, show_depth, rgbWeight, 0
                )
                # cv2.imshow('blended', blended)
                # cv2.imshow('depth', show_depth)
                vis_base = blended
            # Save frame if user presses the 's' key
            key = cv2.waitKey(1)
            if saveFrame is not None and key & 0xFF == ord("s"):
                logging.info("----------------- Saving Frame")
                saveFrame(color, depth, center_pose, bbox, opts)
            if opts.simpublish_ip is not None and key & 0xFF == ord("v"):
                send_vr(color, depth, opts, bbox, center_pose)
            # visualize the scene
            if center_pose is not None:
                vis = draw_posed_3d_box(
                    K, img=vis_base, ob_in_cam=center_pose, bbox=bbox
                )
                vis = draw_xyz_axis(
                    vis_base,
                    ob_in_cam=center_pose,
                    scale=0.1,
                    K=K,
                    thickness=3,
                    transparency=0,
                    is_input_rgb=True,
                )
                cv2.imshow("1", vis)
            if key & 0xFF == ord("q"):
                break
        if saveFrame:
            saveFrame(color, depth, center_pose, bbox, opts)

    return run


# Precompute the model with run_ycbv/run_linemode in run_nerf.py
def run_live_capture(opts, get_mask=mask, device="cuda:0", saveFrame=None):
    """
    Create a method to run the estimator with a pipeline, either live or a recorded stream
    """

    def run(pipeline, depth_scale=4.0):

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
        # mask = np.ones((intrinsics.height, intrinsics.width))

        rgbWeight = 0.75

        align_to = rs.stream.color
        align = rs.align(align_to)

        while True:
            # Getting a frame from the realsense camera
            raw_frames = pipeline.wait_for_frames()

            frames = align.process(raw_frames)
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            # Convert images to numpy arrays and apply necessary transformations
            raw_depth = np.asanyarray(depth_frame.get_data())
            # depth = raw_depth / np.max(raw_depth) * 4.0
            depth = raw_depth * depth_scale

            color = np.uint8(color_frame.get_data())

            cv2.imshow("color", color)
            vis_base = color
            if opts.show_depth:
                if len(depth.shape) < 3:
                    show_depth = cv2.applyColorMap(
                        cv2.convertScaleAbs(raw_depth, alpha=0.1, beta=10),
                        cv2.COLORMAP_JET,
                    )
                blended = cv2.addWeighted(
                    color, 1 - rgbWeight, show_depth, rgbWeight, 0
                )
                # cv2.imshow('blended', blended)
                # cv2.imshow('depth', show_depth)
                vis_base = blended
            # Save frame if user presses the 's' key
            key = cv2.waitKey(1)
            if saveFrame is not None and key & 0xFF == ord("s"):
                logging.info("----------------- Saving Frame")
                saveFrame(color, depth, None, None, opts)
            if opts.simpublish_ip is not None and key & 0xFF == ord("v"):
                send_vr(color, depth, opts)
            # visualize the scene
            cv2.imshow("1", vis_base)
            if key & 0xFF == ord("q"):
                break

    return run


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_view_dir", type=str, default="assets/YCBV/ref_views_16")
    parser.add_argument("--recorded_path", type=str, default=None)
    parser.add_argument("--save_to", type=str, default="test")
    parser.add_argument("--ob_id", type=int, default=None)
    parser.add_argument("--debug", type=int, default=0)
    parser.add_argument("--debug_dir", type=str, default=f"{code_dir}/debug")
    parser.add_argument("--track_refine_iter", type=int, default=2)
    parser.add_argument("--start_pose_path", type=str, default=None)
    parser.add_argument("--ob_name", type=str, default=None)
    parser.add_argument("-c", "--capture", action="store_true")
    parser.add_argument("-i", "--simpublish_ip", default=None)
    parser.add_argument("--demo", default=False)
    parser.add_argument("--show_depth", default=False, action="store_true")
    opts = parser.parse_args()
    if opts.ob_name is None:
        opts.ob_name = opts.ob_id
    if opts.simpublish_ip is not None:
        import vr
    if opts.ob_id is None:
        run_with_pipeline(run_live_capture(opts, saveFrame=save_frame), opts)
    elif opts.demo:
        run_with_pipeline(run_demo, opts)
    else:
        if opts.recorded_path:
            run_with_bag(run_live_estimation(opts), opts.recorded_path)
        else:
            run_with_pipeline(run_live_estimation(opts, saveFrame=save_frame), opts)

# Example Usages:
# install SimPublisher in the docker container:
"""
cd SimPublisher && pip install -e . && cd ..
"""
"""
python run_realsense.py --show_depth --ob_id 14 -i 10.10.10.220
python -m debugpy --wait-for-client --listen 0.0.0.0:5678 run_realsense.py --show_depth --ob_id 10 -i 10.10.10.220
"""
