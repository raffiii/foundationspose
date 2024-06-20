import pyrealsense2 as rs
import numpy as np
import cv2
import trimesh
import argparse

from estimater import *


def run_pose_estimation_worker(color, depth,K, est:FoundationPose, ob_mask=None, debug=0, ob_id=None, device='cuda:0'):
  torch.cuda.set_device(device)
  est.to_device(device)
  est.glctx = dr.RasterizeCudaContext(device=device)

  result = NestDict()

  H,W = color.shape[:2]

  debug_dir =est.debug_dir

  if ob_mask is None:
    pose = est.register(K=K, rgb=color, depth=depth, ob_mask=ob_mask)
    logging.info(f"pose:\n{pose}")
  else:
    pose = est.track_one(rgb=color, depth=depth, K=K, iteration=args.track_refine_iter)

  return pose


def get_reconstructed_mesh(ob_id, ref_view_dir):
  mesh = trimesh.load(os.path.abspath(f'{ref_view_dir}/ob_{ob_id:07d}/model/model.obj'))
  return mesh



def build_estimator(mesh,opts):

    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)

    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer, refiner=refiner, debug_dir=opts.debug_dir, debug=opts.debug, glctx=glctx)
    logging.info("estimator initialization done")
    return est, to_origin, bbox, 


def run_with_pipeline(function):
    print("reset start")
    ctx = rs.context()
    devices = ctx.query_devices()
    for dev in devices:
        dev.hardware_reset()
    print("reset done")

    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()

    # Start streaming from the default RealSense camera
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    pipeline.start(config)
    result = None
    try:
        result = function(pipeline)
    finally:
        # Stop streaming
        pipeline.stop()
        cv2.destroyAllWindows()
    return result

def run_demo(pipeline):
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
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # Stack both images horizontally
        images = np.hstack((color_image, depth_colormap))

        # Show images
        cv2.imshow('RealSense', images)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


# Precompute the model with run_ycbv/run_linemode in run_nerf.py
def run_live_estimation(opts):
    mesh = get_reconstructed_mesh(opts.ob_id, opts.ref_view_dir)
    est, to_origin, bbox = build_estimator(mesh,opts)
    def run(pipeline):
        mask = capture_mask(pipeline)
        shape = mask.shape
        mask = fix_mask(mask, np.vstack((np.zeros((shape[0]//2,shape[1])),np.ones((shape[0]-shape[0]//2,shape[1])))))
    
        K = pipeline.todo
        while(True):
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            mask = np.ones(color_frame.shape()[:2])
            run_pose_estimation_worker(color_image, depth_image, K, est,ob_mask=mask)   
    return run     
        
def compute_mask(object, background, threshold=0.05):
    alpha = 1+threshold
    result = np.hstack(((background > alpha * object) * 1.0 , (object < alpha* background) * 1.0))
    cv2.imshow('RealSense', result * 1.0 )
    cv2.waitKey(0)   
    cv2.destroyAllWindows() 
    less,more = (background > alpha * object , object <  alpha* background) 
    diff = (np.max(more,axis=2) * 1.0 + np.max(less,axis=2) * 1.0) > 0
    return diff
    
        
def capture_mask(pipeline):
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    background = np.asanyarray(color_frame.get_data())
    input("Place object to track and hit enter:")
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    object_image = np.asanyarray(color_frame.get_data())
    return compute_mask(object_image,background)

def fix_mask(mask,maskmask):
    return (mask * 1.0) * (maskmask * 1.0)

def show_mask():
    result = run_with_pipeline(capture_mask)
    shape = result.shape
    result = fix_mask(result, np.vstack((np.zeros((shape[0]//2,shape[1])),np.ones((shape[0]-shape[0]//2,shape[1])))))
    cv2.imshow('RealSense', result * 1.0 )
    cv2.waitKey(0)   
    cv2.destroyAllWindows() 

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref_view_dir', type=str, default='assets/banana/ref_views_16')
    parser.add_argument('--ob_id', type=int,default=10)
    parser.add_argument('--debug', type=int, default=0)
    parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug')
    opts = parser.parse_args()
    run_with_pipeline(run_live_estimation(opts))