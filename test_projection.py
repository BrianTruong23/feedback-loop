import robosuite as suite
from robosuite.utils.camera_utils import get_camera_transform_matrix, get_real_depth_map, transform_from_pixels_to_world
import numpy as np
env = suite.make("PickPlace", robots="Panda", has_renderer=False, use_camera_obs=True, camera_names="frontview", camera_heights=256, camera_widths=256, camera_depths=True, has_offscreen_renderer=True)
obs = env.reset()
cam_mat = get_camera_transform_matrix(env.sim, "frontview", 256, 256)
pixels = np.array([128, 128])
pt_raw = transform_from_pixels_to_world(pixels, get_real_depth_map(env.sim, obs["frontview_depth"]), cam_mat)
print("NO INV, pixels=[u,v]:", pt_raw)

pt_inv_uv = transform_from_pixels_to_world(pixels, get_real_depth_map(env.sim, obs["frontview_depth"]), np.linalg.inv(cam_mat))
print("INV, pixels=[u,v]:", pt_inv_uv)

pt_inv_vu = transform_from_pixels_to_world(np.array([128, 128]), get_real_depth_map(env.sim, obs["frontview_depth"]), np.linalg.inv(cam_mat))
print("INV, pixels=[v,u]:", pt_inv_vu)

pt_raw_vu = transform_from_pixels_to_world(np.array([128, 128]), get_real_depth_map(env.sim, obs["frontview_depth"]), cam_mat)
print("NO INV, pixels=[v,u]:", pt_raw_vu)

print("ROSTER TARGET EEF:", obs['robot0_eef_pos'])

