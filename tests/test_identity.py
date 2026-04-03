import robosuite as suite
from robosuite.utils.camera_utils import get_camera_transform_matrix, get_real_depth_map, transform_from_pixels_to_world, project_points_from_world_to_camera
import numpy as np
import warnings
warnings.filterwarnings('ignore')

env = suite.make("PickPlace", robots="Panda", has_renderer=False, use_camera_obs=True, camera_names="frontview", camera_heights=256, camera_widths=256, camera_depths=True, has_offscreen_renderer=True)
obs = env.reset()

cam_mat = get_camera_transform_matrix(env.sim, "frontview", 256, 256)
cam_mat_inv = np.linalg.inv(cam_mat)

# 1. Pick a point on the table
world_pt = np.array([0.15, 0.05, 0.81])

# 2. Project World to Pixel
pixel = project_points_from_world_to_camera(world_pt, cam_mat, 256, 256)
print("Projected pixel [v, u]:", pixel)

v = pixel[0]
u = pixel[1]

# What is depth at this pixel?
depth = get_real_depth_map(env.sim, obs["frontview_depth"])
z_val = depth[int(v), int(u)]
print("Depth value at v,u:", z_val)

# Is the Depth flipped or not?! Let's test the depth map orientation
# Provide the EXACT pixel from the projection to transform_from_pixels_to_world
world_pt_recovered = transform_from_pixels_to_world(np.array([pixel]), depth, cam_mat_inv)
print("Recovered World Pt with RAW depth:", world_pt_recovered)

# What if depth map was flipped?
world_pt_recovered_flipped = transform_from_pixels_to_world(np.array([pixel]), depth[::-1], cam_mat_inv)
print("Recovered World Pt with FLIPPED depth:", world_pt_recovered_flipped)

# Print Actual object positions just to verify their bounding boxes
for obj in env.model.mujoco_objects:
    print(obj.name, env.sim.data.body_xpos[env.sim.model.body_name2id(obj.root_body)])

