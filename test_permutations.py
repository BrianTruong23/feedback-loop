import robosuite as suite
from robosuite.utils.camera_utils import get_camera_transform_matrix, get_real_depth_map, transform_from_pixels_to_world
import numpy as np

env = suite.make("PickPlace", robots="Panda", has_renderer=False, use_camera_obs=True, camera_names="frontview", camera_heights=256, camera_widths=256, camera_depths=True, has_offscreen_renderer=True)
obs = env.reset()

cam_mat_inv = np.linalg.inv(get_camera_transform_matrix(env.sim, "frontview", 256, 256))
depth = get_real_depth_map(env.sim, obs["frontview_depth"])

u = 120
v = 150 # In the flipped CV-style image, this is on the table
orig_v = 255 - v

print("u =", u, "v = ", v, "orig_v = ", orig_v)

print("1. [u, v]:", transform_from_pixels_to_world(np.array([u, v]), depth, cam_mat_inv))
print("2. [v, u]:", transform_from_pixels_to_world(np.array([v, u]), depth, cam_mat_inv))
print("3. [u, orig_v]:", transform_from_pixels_to_world(np.array([u, orig_v]), depth, cam_mat_inv))
print("4. [orig_v, u]:", transform_from_pixels_to_world(np.array([orig_v, u]), depth, cam_mat_inv))

