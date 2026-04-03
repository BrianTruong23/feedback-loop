import robosuite as suite
from robosuite.utils.camera_utils import get_camera_transform_matrix, get_real_depth_map, transform_from_pixels_to_world
import numpy as np
import warnings
warnings.filterwarnings('ignore')

env = suite.make("PickPlace", robots="Panda", has_renderer=False, use_camera_obs=True, camera_names="frontview", camera_heights=256, camera_widths=256, camera_depths=True, has_offscreen_renderer=True)
obs = env.reset()
cam_mat_inv = np.linalg.inv(get_camera_transform_matrix(env.sim, "frontview", 256, 256))

u = 120
v = 150 # Top-left origin, roughly milk center in typical spawn
orig_v = 255 - v

depth = get_real_depth_map(env.sim, obs["frontview_depth"])

# Original Approach before any tinkering: [u, orig_v] AND NO INVERSE! Wait, no inverse was in Attempt 1. Let's test it with NO INVERSE!
cam_mat_no_inv = get_camera_transform_matrix(env.sim, "frontview", 256, 256)
pt_3d_orig_noinv = transform_from_pixels_to_world(np.array([u, orig_v]), depth, cam_mat_no_inv)
print(f"[u, orig_v], NO INV: {pt_3d_orig_noinv}")

pt_3d_orig_inv = transform_from_pixels_to_world(np.array([u, orig_v]), depth, cam_mat_inv)
print(f"[u, orig_v], WITH INV: {pt_3d_orig_inv}")

# What if standard [orig_v, u] was correct?
pt_3d_vu_inv = transform_from_pixels_to_world(np.array([orig_v, u]), depth, cam_mat_inv)
print(f"[orig_v, u], WITH INV: {pt_3d_vu_inv}")

