"""
manual_grasp.py — Interactive coordinate grasp tester.

Usage:
    python manual_grasp.py                    # prompts for x, y interactively
    python manual_grasp.py --x 0.15 --y -0.15 # single shot

The robot will:
  1. Set up the simplified 1-bin / 2-object scene (Cereal + Milk)
  2. Show the initial composite view (Front + Bird) and save it
  3. Execute a grasp at the supplied world (x, y) coordinates
  4. Show the moment-of-contact composite view
  5. Report whether the grasp succeeded, then loop back for another attempt
"""

import argparse
import os
import datetime
import numpy as np
import robosuite as suite
from PIL import Image
from robosuite.utils.camera_utils import get_camera_transform_matrix, get_real_depth_map, transform_from_pixels_to_world
import robosuite.utils.transform_utils as T

IMG_H, IMG_W = 512, 512


# ── helpers borrowed/adapted from baseline.py ────────────────────────────────

def simplify_environment(env):
    for obj_name in ["Bread_joint0", "Can_joint0"]:
        j_id = env.sim.model.joint_name2id(obj_name)
        q_idx = env.sim.model.jnt_qposadr[j_id]
        env.sim.data.qpos[q_idx : q_idx + 3] = np.array([5.0, 5.0, -1.0])

    cereal_id = env.sim.model.joint_name2id("Cereal_joint0")
    cereal_idx = env.sim.model.jnt_qposadr[cereal_id]
    env.sim.data.qpos[cereal_idx : cereal_idx + 3] = np.array([0.15, -0.15, 0.9])

    milk_id = env.sim.model.joint_name2id("Milk_joint0")
    milk_idx = env.sim.model.jnt_qposadr[milk_id]
    env.sim.data.qpos[milk_idx : milk_idx + 3] = np.array([0.05, -0.30, 0.9])

    env.sim.forward()


def draw_red_grid_on_array(img_array):
    from PIL import ImageDraw
    image = Image.fromarray(img_array.copy())
    draw = ImageDraw.Draw(image)
    width, height = image.size
    step = 32
    for x in range(0, width, step):
        draw.line([(x, 0), (x, height)], fill=(255, 0, 0), width=1)
        draw.text((x + 2, 2), str(x), fill=(255, 255, 255))
    for y in range(0, height, step):
        draw.line([(0, y), (width, y)], fill=(255, 0, 0), width=1)
        draw.text((2, y + 2), str(y), fill=(255, 255, 255))
    return np.array(image)


def draw_gripper_gizmo(img_array, obs, env, camera_name):
    from PIL import ImageDraw
    from robosuite.utils.camera_utils import project_points_from_world_to_camera
    try:
        image = Image.fromarray(img_array)
        draw = ImageDraw.Draw(image)
        h, w = img_array.shape[:2]
        eef_pos = obs['robot0_eef_pos']
        eef_quat = obs['robot0_eef_quat']
        eef_rot = T.quat2mat(eef_quat)
        cam_transform = get_camera_transform_matrix(env.sim, camera_name, h, w)
        px_center = project_points_from_world_to_camera(eef_pos.reshape(1, 3), cam_transform, h, w)[0]
        c_y, c_x = px_center[0], px_center[1]
        axis_len = 0.1
        x_end = eef_pos + eef_rot[:, 0] * axis_len
        px_x = project_points_from_world_to_camera(x_end.reshape(1, 3), cam_transform, h, w)[0]
        y_end = eef_pos + eef_rot[:, 1] * axis_len
        px_y = project_points_from_world_to_camera(y_end.reshape(1, 3), cam_transform, h, w)[0]
        draw.line([(c_x, c_y), (px_x[1], px_x[0])], fill=(255, 0, 0), width=3)
        draw.line([(c_x, c_y), (px_y[1], px_y[0])], fill=(0, 255, 0), width=3)
        return np.array(image)
    except Exception as e:
        print(f"Gizmo error: {e}")
        return img_array


def create_composite_image(obs, env):
    front = obs["frontview_image"][::-1]
    bird = obs["birdview_image"][::-1]
    front = draw_gripper_gizmo(draw_red_grid_on_array(front), obs, env, "frontview")
    bird = draw_gripper_gizmo(draw_red_grid_on_array(bird), obs, env, "birdview")
    return np.hstack([front, bird])


def step_towards(obs, env, target_xyz, gripper_action, steps=40):
    pos_gain = 2.2
    max_cart_action = 0.20
    pos_tol = 0.005
    start_xyz = obs['robot0_eef_pos'].copy()
    target_xyz = np.array(target_xyz, dtype=float)

    for step_idx in range(steps):
        progress = (step_idx + 1) / steps
        eased_progress = 0.5 - 0.5 * np.cos(np.pi * progress)
        waypoint = start_xyz + (target_xyz - start_xyz) * eased_progress
        current_eef = obs['robot0_eef_pos']
        delta = waypoint - current_eef
        if np.linalg.norm(target_xyz - current_eef) < pos_tol:
            break
        action = np.zeros(7)
        action[:3] = np.clip(delta * pos_gain, -max_cart_action, max_cart_action)
        action[6] = gripper_action
        obs, _, _, _ = env.step(action)

    settle_action = np.zeros(7)
    settle_action[6] = gripper_action
    obs, _, _, _ = env.step(settle_action)
    return obs


def retract_arm(obs, env):
    retract_pos = np.array([0.4, -0.6, 1.4])
    return step_towards(obs, env, retract_pos, gripper_action=-1, steps=40)


def check_objects_lifted(env):
    results = {}
    for o in env.objects:
        body_id = env.sim.model.body_name2id(o.root_body)
        z_pos = env.sim.data.body_xpos[body_id][2]
        results[o.name] = z_pos
    return results


def print_ground_truth(env):
    body_map = {"Cereal": "Cereal_main", "Milk": "Milk_main"}
    print("\n  Ground Truth Object Positions:")
    for name, body in body_map.items():
        try:
            pos = env.sim.data.get_body_xpos(body)
            print(f"    {name}: x={pos[0]:.4f}  y={pos[1]:.4f}  z={pos[2]:.4f}")
        except Exception:
            pass


# ── main session ─────────────────────────────────────────────────────────────

def run_session(initial_x=None, initial_y=None):
    ts = datetime.datetime.now().strftime("%m%d_%I_%M_%p").lower()
    run_dir = os.path.join("runs", f"manual_{ts}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"\nRun directory: {run_dir}")

    print("Initialising Robosuite PickPlace environment...")
    env = suite.make(
        env_name="PickPlace",
        robots="Panda",
        controller_configs=suite.load_composite_controller_config(controller="BASIC"),
        has_renderer=False,
        use_camera_obs=True,
        camera_names=["frontview", "birdview"],
        camera_heights=IMG_H,
        camera_widths=IMG_W,
        camera_depths=True,
        has_offscreen_renderer=True,
        control_freq=20,
        render_camera="frontview",
    )

    obs = env.reset()
    simplify_environment(env)
    obs = retract_arm(obs, env)

    # Save + show initial composite
    initial_composite = create_composite_image(obs, env)
    initial_path = os.path.join(run_dir, "initial_composite.png")
    Image.fromarray(initial_composite).save(initial_path)
    print(f"\nInitial composite saved → {initial_path}")
    print_ground_truth(env)

    attempt = 0
    x_in, y_in = initial_x, initial_y

    while True:
        attempt += 1
        print(f"\n{'='*55}")
        print(f" Attempt {attempt}")
        print(f"{'='*55}")

        if x_in is None or y_in is None:
            try:
                raw = input("\nEnter world coordinates  x y  (or 'q' to quit): ").strip()
                if raw.lower() in ("q", "quit", "exit"):
                    break
                parts = raw.split()
                x_in = float(parts[0])
                y_in = float(parts[1])
            except (ValueError, IndexError):
                print("  Invalid input. Try e.g.:  0.15 -0.15")
                x_in = y_in = None
                continue

        z_grasp = 0.825  # hardcoded table level (same as baseline)
        target_xy = np.array([x_in, y_in, z_grasp])

        print(f"\n  Targeting:  x={x_in:.4f}  y={y_in:.4f}  z={z_grasp}")
        print_ground_truth(env)

        # Hover above
        hover = target_xy.copy()
        hover[2] += 0.2
        print("  → Hovering...")
        obs = step_towards(obs, env, hover, gripper_action=-1, steps=40)

        # Lower to grasp
        print("  → Lowering...")
        obs = step_towards(obs, env, target_xy, gripper_action=-1, steps=30)

        # Close gripper
        print("  → Grasping...")
        obs = step_towards(obs, env, target_xy, gripper_action=1, steps=20)

        # Capture moment of contact
        mid_composite = create_composite_image(obs, env)
        mid_path = os.path.join(run_dir, f"attempt_{attempt}_contact.png")
        Image.fromarray(mid_composite).save(mid_path)
        print(f"  Contact composite saved → {mid_path}")

        # Lift
        print("  → Lifting...")
        lift = target_xy.copy()
        lift[2] += 0.3
        obs = step_towards(obs, env, lift, gripper_action=1, steps=50)

        # Check result
        heights = check_objects_lifted(env)
        print("\n  Object heights after lift:")
        for name, z in heights.items():
            lifted = " ← LIFTED" if z > 0.95 else ""
            print(f"    {name}: z={z:.4f}{lifted}")

        lifted_names = [n for n, z in heights.items() if z > 0.95]
        if lifted_names:
            print(f"\n  Result: LIFTED → {lifted_names}")
        else:
            print("\n  Result: NOTHING LIFTED")

        # Retract for next attempt
        print("  → Retracting arm...")
        obs = retract_arm(obs, env)

        # Capture post-retract composite
        after_composite = create_composite_image(obs, env)
        after_path = os.path.join(run_dir, f"attempt_{attempt}_after.png")
        Image.fromarray(after_composite).save(after_path)
        print(f"  After-retract composite saved → {after_path}")

        # Reset for next loop (prompt again)
        x_in = y_in = None

    env.close()
    print(f"\nSession complete. All files in: {run_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive manual grasp tester")
    parser.add_argument("--x", type=float, default=None, help="World X coordinate for grasp")
    parser.add_argument("--y", type=float, default=None, help="World Y coordinate for grasp")
    args = parser.parse_args()
    run_session(initial_x=args.x, initial_y=args.y)
