from estimater import *
import stream, oakd_stream, realsense_stream
import cv2
import queue
import time

##############################################################################
# Pose estimamtion

def run_pose_estimation_worker(color, depth,K, est:FoundationPose,opts, ob_mask=None, debug=0, ob_id=None, device='cuda:0'):
  logging.info("--------------- Called run worker")
  if ob_mask is not None:
    pose = est.register(K=K, rgb=color, depth=depth, ob_mask=ob_mask)
    logging.info(f"pose:\n{pose}")
  else:
    pose = est.track_one(rgb=color, depth=depth, K=K, iteration=opts.track_refine_iter)

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


##############################################################################
# Mask heuristic


def compute_mask(object, background, threshold=0.05):
    """
    A heuristic to predict the object mask by using the difference between two 
    images, without and with the object
    """
    alpha = 1+threshold
    less,more = (background > alpha * object , object <  alpha* background) 
    diff = (np.max(more,axis=2) * 1.0 + np.max(less,axis=2) * 1.0) > 0
    return diff
    
        
def capture_mask(q, cam_id):
    """
    Capture the two images for the mask heuristic
    """
    logging.info("Capturing mask")
    background = q.get()[cam_id]
    #input('-' * 40 + "Place object and hit enter:")
    #cv2.namedWindow('Object_place', cv2.WINDOW_NORMAL)
    #logging.info("Waiting for object placement")
    cv2.imshow('Object_place', background.color)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # Drop intermediate frames
    while not q.empty():
        q.get()
    foreground = q.get()[cam_id]
    while foreground.id != background.id:
        foreground = q.get()
    return compute_mask(foreground.color,background.color)

def fix_mask(mask,maskmask):
    """
    Fix the mask for noise in known background regions
    """
    return (mask * 1.0) * (maskmask * 1.0)

def mask(pipeline,opts):
    """
    Capture and fix a mask using a heuristic
    """
    mask = capture_mask(pipeline)
    shape = mask.shape
    mask = fix_mask(mask, np.vstack((np.zeros((shape[0]//2,shape[1])),np.ones((shape[0]-shape[0]//2,shape[1])))))
    #np.save()
    return mask

##############################################################################
# Runner

def build_runner(opts, get_mask=capture_mask, device='cuda:0'):
    """
    Build a funtion using opts that takes a list of queues to run estimation with
    """
    mesh = get_reconstructed_mesh(opts.ob_id, opts.ref_view_dir)
    est, to_origin, bbox = build_estimator(mesh,opts)

    def run(streams, q, tracker=0):
        tracker = streams[tracker]
        mask = get_mask(q, tracker.cam_id)

        torch.cuda.set_device(device)
        est.to_device(device)
        est.glctx = dr.RasterizeCudaContext(device=device)
        poses = []
        while tracker.running:
            frames = q.get()
            f = frames[tracker.cam_id]
            logging.info(f"Color shape: {f.color.shape} , Depth shape: {f.depth.shape} \n")
            logging.info(f"Intrinsics (K): {[s.K for s in streams]}")
            pose = run_pose_estimation_worker(f.color, f.depth, tracker.K, est,opts,ob_mask=mask) 
            mask=None  
            
            center_pose = pose@np.linalg.inv(to_origin)
            poses += [center_pose]
            logging.info(f"Center pose: \n {center_pose}")
            vis = draw_posed_3d_box(tracker.K, img=f.color, ob_in_cam=center_pose, bbox=bbox)
            vis = draw_xyz_axis(f.color, ob_in_cam=center_pose, scale=0.1, K=tracker.K, thickness=3, transparency=0, is_input_rgb=True)
            cv2.imshow('1', vis[...,::1])
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        np.savez(f"{opts.debug_dir}/{opts.save_to}.npz",mask=mask, poses=np.concatenate(poses,axis=0))
    return run


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref_view_dir', type=str, default='assets/banana/ref_views_16')
    parser.add_argument('--recorded_path', type=str,default=None)
    parser.add_argument('--save_to',type=str,default='test')
    parser.add_argument('--ob_id', type=int,default=10)
    parser.add_argument('--debug', type=int, default=0)
    parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug')
    parser.add_argument('--track_refine_iter', type=int, default=2)
    parser.add_argument('--start_pose_path', type=str, default=None)
    opts = parser.parse_args()
    stream.multistream_sync(build_runner(opts), oakd_stream.available_streams(1))
