"""
Interactive 2D→3D projection tester.

Projects your pixel delta to a 3D world target, moves the arm there,
saves a debug image (Gemini-style: grid + gizmo + target markers) and
a short MP4 — all in test_projection_output/ — then opens both.

Usage examples:
  python test_delta_projection.py --delta-u 23 --delta-v -25
  python test_delta_projection.py --base-u 172 --base-v 280 --delta-u 23 --delta-v -25
  python test_delta_projection.py --base-u 195 --base-v 305 --delta-u 0 --delta-v 0

Arguments:
  --base-u   Pixel column of the last grasp target (default: 172)
  --base-v   Pixel row    of the last grasp target (default: 280)
  --delta-u  Pixel shift in u (column) suggested by Gemini (default: 0)
  --delta-v  Pixel shift in v (row)    suggested by Gemini (default: 0)
  --grasp-z  Fixed grasp plane height in world-space metres (default: 0.825)
"""

import argparse
import os
import subprocess
import numpy as np
import imageio
from PIL import Image, ImageDraw
import robosuite as suite
import robosuite.utils.transform_utils as T
from robosuite.utils.camera_utils import (
    get_camera_transform_matrix,
    get_real_depth_map,
    transform_from_pixels_to_world,
    project_points_from_world_to_camera,
)

# ── constants ─────────────────────────────────────────────────────────────────
IMG_H, IMG_W   = 512, 512
CAMERA         = "frontview"
GRASP_Z_DEFAULT = 0.825
OUTPUT_DIR     = "test_projection_output"


# ── drawing helpers (mirrored from baseline.py) ───────────────────────────────

def draw_red_grid_on_array(img_array):
    image = Image.fromarray(img_array.copy())
    draw  = ImageDraw.Draw(image)
    w, h  = image.size
    step  = 32
    for x in range(0, w, step):
        draw.line([(x, 0), (x, h)], fill=(255, 0, 0), width=1)
        draw.text((x + 2, 2), str(x), fill=(255, 255, 255))
    for y in range(0, h, step):
        draw.line([(0, y), (w, y)], fill=(255, 0, 0), width=1)
        draw.text((2, y + 2), str(y), fill=(255, 255, 255))
    return np.array(image)


def draw_gripper_gizmo(img_array, obs, env, camera_name):
    try:
        image = Image.fromarray(img_array)
        draw  = ImageDraw.Draw(image)
        h, w  = img_array.shape[:2]

        eef_pos  = obs["robot0_eef_pos"]
        eef_quat = obs["robot0_eef_quat"]
        eef_rot  = T.quat2mat(eef_quat)

        cam_transform = get_camera_transform_matrix(env.sim, camera_name, h, w)
        px_center     = project_points_from_world_to_camera(
            eef_pos.reshape(1, 3), cam_transform, h, w
        )[0]
        c_y, c_x = px_center[0], px_center[1]

        axis_len = 0.1
        x_end = eef_pos + eef_rot[:, 0] * axis_len
        px_x  = project_points_from_world_to_camera(x_end.reshape(1, 3), cam_transform, h, w)[0]
        y_end = eef_pos + eef_rot[:, 1] * axis_len
        px_y  = project_points_from_world_to_camera(y_end.reshape(1, 3), cam_transform, h, w)[0]

        draw.line([(c_x, c_y), (px_x[1], px_x[0])], fill=(255, 0, 0),   width=3)
        draw.line([(c_x, c_y), (px_y[1], px_y[0])], fill=(0, 255, 0),   width=3)
        return np.array(image)
    except Exception as e:
        print(f"  Gizmo draw skipped: {e}")
        return img_array


def draw_last_target(img_array, u, v):
    """Cyan crosshair — the base pixel (last commanded target)."""
    image  = Image.fromarray(img_array.copy())
    draw   = ImageDraw.Draw(image)
    h, w   = img_array.shape[:2]
    ax, ay = int(np.clip(round(u), 0, w - 1)), int(np.clip(round(v), 0, h - 1))
    cl     = 10
    color  = (0, 255, 255)
    draw.line([(ax - cl, ay), (ax + cl, ay)], fill=color, width=3)
    draw.line([(ax, ay - cl), (ax, ay + cl)], fill=color, width=3)
    draw.ellipse([(ax - 4, ay - 4), (ax + 4, ay + 4)], outline=color, width=2)
    draw.text((ax + 8, max(ay - 18, 0)), f"LAST TARGET  u={u:.0f} v={v:.0f}", fill=color)
    return np.array(image)


def draw_corrected_target(img_array, last_u, last_v, new_u, new_v):
    """Orange arrow + circle — the Gemini-corrected pixel."""
    image   = Image.fromarray(img_array.copy())
    draw    = ImageDraw.Draw(image)
    h, w    = img_array.shape[:2]
    sx, sy  = int(np.clip(round(last_u), 0, w - 1)), int(np.clip(round(last_v), 0, h - 1))
    px, py  = int(np.clip(round(new_u),  0, w - 1)), int(np.clip(round(new_v),  0, h - 1))
    color   = (255, 64, 0)
    draw.line([(sx, sy), (px, py)], fill=color, width=3)
    draw.ellipse([(px - 6, py - 6), (px + 6, py + 6)], outline=color, width=3)
    draw.text((px + 8, max(py - 18, 0)), f"CORRECTED  u={new_u:.0f} v={new_v:.0f}", fill=color)
    return np.array(image)


# ── projection helpers (mirrored from baseline.py) ───────────────────────────

def robust_project_front_pixel_to_world(
    real_depth, cam_mat, img_h, img_w, u_px, v_px,
    reference_world=None, search_radius=12, stride=2
):
    candidates = []
    center_u = float(np.clip(u_px, 0, img_w - 1))
    center_v = float(np.clip(v_px, 0, img_h - 1))

    for dv in range(-search_radius, search_radius + 1, stride):
        for du in range(-search_radius, search_radius + 1, stride):
            sample_u = float(np.clip(center_u + du, 0, img_w - 1))
            sample_v = float(np.clip(center_v + dv, 0, img_h - 1))
            sample_orig_v  = (img_h - 1) - sample_v
            sample_pixels  = np.array([sample_orig_v, sample_u])
            try:
                world_pt = transform_from_pixels_to_world(sample_pixels, real_depth, cam_mat)
            except Exception:
                continue
            if not np.all(np.isfinite(world_pt)):
                continue
            pixel_dist      = np.hypot(du, dv)
            world_xy_penalty = 0.0
            if reference_world is not None:
                world_xy_penalty = np.linalg.norm(world_pt[:2] - reference_world[:2]) * 250.0
            score = pixel_dist + world_xy_penalty
            candidates.append((score, pixel_dist, world_pt, sample_u, sample_v))

    if not candidates:
        fallback_orig_v = (img_h - 1) - center_v
        return (
            transform_from_pixels_to_world(
                np.array([fallback_orig_v, center_u]), real_depth, cam_mat
            ),
            center_u, center_v,
        )

    candidates.sort(key=lambda item: item[0])
    top_world    = np.array([item[2] for item in candidates[: min(9, len(candidates))]])
    median_world = np.median(top_world, axis=0)
    best = min(
        candidates,
        key=lambda item: np.linalg.norm(item[2][:2] - median_world[:2]) + item[1] * 0.1,
    )
    return best[2], best[3], best[4]


def apply_decoupled_pixel_update(
    real_depth, cam_mat, img_h, img_w,
    last_u, last_v, new_u, new_v, reference_world
):
    base_world, base_used_u, base_used_v = robust_project_front_pixel_to_world(
        real_depth, cam_mat, img_h, img_w, last_u, last_v, reference_world=reference_world
    )
    u_world, used_u, _ = robust_project_front_pixel_to_world(
        real_depth, cam_mat, img_h, img_w, new_u, last_v, reference_world=reference_world
    )
    v_world, _, used_v = robust_project_front_pixel_to_world(
        real_depth, cam_mat, img_h, img_w, last_u, new_v, reference_world=reference_world
    )

    delta_u_world = u_world[:2] - base_world[:2]
    delta_v_world = v_world[:2] - base_world[:2]

    u_axis = 0 if abs(delta_u_world[0]) >= abs(delta_u_world[1]) else 1
    v_axis = 1 - u_axis

    new_world         = reference_world.copy()
    new_world[u_axis] = reference_world[u_axis] + delta_u_world[u_axis]
    new_world[v_axis] = reference_world[v_axis] + delta_v_world[v_axis]

    return new_world, {
        "base_world":      base_world,
        "base_used_u":     base_used_u,
        "base_used_v":     base_used_v,
        "used_u":          used_u,
        "used_v":          used_v,
        "u_axis":          u_axis,
        "v_axis":          v_axis,
        "delta_u_world_x": float(delta_u_world[0]),
        "delta_u_world_y": float(delta_u_world[1]),
        "delta_v_world_x": float(delta_v_world[0]),
        "delta_v_world_y": float(delta_v_world[1]),
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Test 2D pixel delta → 3D world-target projection"
    )
    parser.add_argument("--base-u",  type=float, default=172.0)
    parser.add_argument("--base-v",  type=float, default=280.0)
    parser.add_argument("--delta-u", type=float, default=0.0)
    parser.add_argument("--delta-v", type=float, default=0.0)
    parser.add_argument("--grasp-z", type=float, default=GRASP_Z_DEFAULT)
    args = parser.parse_args()

    last_u = args.base_u
    last_v = args.base_v
    new_u  = last_u + args.delta_u
    new_v  = last_v + args.delta_v

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    video_path = os.path.join(OUTPUT_DIR, "movement.mp4")
    debug_path = os.path.join(OUTPUT_DIR, "debug_view.png")

    print("\n=== test_delta_projection ===")
    print(f"  Base pixel       : u={last_u:.1f}, v={last_v:.1f}")
    print(f"  Delta            : Δu={args.delta_u:+.1f}, Δv={args.delta_v:+.1f}")
    print(f"  Corrected pixel  : u={new_u:.1f}, v={new_v:.1f}")
    print(f"  Grasp Z          : {args.grasp_z} m")
    print(f"  Output dir       : {OUTPUT_DIR}/")
    print()

    # ── environment setup ─────────────────────────────────────────────────────
    print("Initialising simulation…")
    env = suite.make(
        env_name="PickPlace",
        robots="Panda",
        controller_configs=suite.load_composite_controller_config(controller="BASIC"),
        has_renderer=False,
        use_camera_obs=True,
        camera_names=[CAMERA],
        camera_heights=IMG_H,
        camera_widths=IMG_W,
        camera_depths=True,
        has_offscreen_renderer=True,
        control_freq=20,
        render_camera=CAMERA,
    )

    obs = env.reset()

    # Place Cereal at canonical position and let physics settle (mirrors simplify_environment)
    cereal_id  = env.sim.model.joint_name2id("Cereal_joint0")
    cereal_idx = env.sim.model.jnt_qposadr[cereal_id]
    env.sim.data.qpos[cereal_idx : cereal_idx + 3] = np.array([0.08, -0.25, 0.9])
    env.sim.forward()
    for _ in range(50):
        env.sim.step()
    obs, *_ = env.step(np.zeros(7))

    # Retract arm first so depth is captured from the same clean view as baseline
    print("Retracting arm for clean depth capture…")
    HOME_POS_EARLY = np.array([0.4, -0.6, 1.4])
    for _ in range(12):
        eef   = obs["robot0_eef_pos"]
        delta = HOME_POS_EARLY - eef
        if np.linalg.norm(delta) < 0.005:
            break
        action     = np.zeros(7)
        action[:3] = np.clip(delta * 3.4, -0.45, 0.45)
        action[6]  = -1
        obs, *_ = env.step(action)

    # ── depth + camera matrix (captured from retracted clean view) ────────────
    real_depth = get_real_depth_map(env.sim, obs[f"{CAMERA}_depth"])
    cam_mat    = np.linalg.inv(get_camera_transform_matrix(env.sim, CAMERA, IMG_H, IMG_W))

    # ── project base pixel → reference world pos ─────────────────────────────
    base_orig_v = (IMG_H - 1) - last_v
    base_3d     = transform_from_pixels_to_world(
        np.array([base_orig_v, last_u]), real_depth, cam_mat
    )
    base_3d[2]      = args.grasp_z
    reference_world = base_3d.copy()
    print(f"  Base pixel → world  : x={base_3d[0]:.4f}, y={base_3d[1]:.4f}, z={base_3d[2]:.4f}")

    # ── apply pixel correction → target world pos ─────────────────────────────
    if args.delta_u == 0.0 and args.delta_v == 0.0:
        print("  Δu=0 and Δv=0 → targeting base pixel directly.")
        target_world = reference_world
        info = {}
    else:
        target_world, info = apply_decoupled_pixel_update(
            real_depth, cam_mat, IMG_H, IMG_W,
            last_u, last_v, new_u, new_v, reference_world
        )
        target_world[2] = args.grasp_z

    # ── print results ─────────────────────────────────────────────────────────
    print()
    print("=" * 50)
    print("  ARM TARGET (world coordinates)")
    print("=" * 50)
    print(f"  X  = {target_world[0]:+.4f} m   (left/right)")
    print(f"  Y  = {target_world[1]:+.4f} m   (forward/back)")
    print(f"  Z  = {target_world[2]:+.4f} m   (up/down, fixed)")
    if info:
        print()
        print("  ── Projection internals ──────────────────────")
        print(f"  Base pixel actually used : u={info['base_used_u']:.1f}, v={info['base_used_v']:.1f}")
        print(f"  Base world               : {info['base_world'][:3]}")
        print(f"  u shifts world axis      : {'X' if info['u_axis']==0 else 'Y'}")
        print(f"  v shifts world axis      : {'X' if info['v_axis']==0 else 'Y'}")
        print(f"  Δu → world [{info['u_axis']}]            : {info['delta_u_world_x']:+.6f} x,  {info['delta_u_world_y']:+.6f} y")
        print(f"  Δv → world [{info['v_axis']}]            : {info['delta_v_world_x']:+.6f} x,  {info['delta_v_world_y']:+.6f} y")
    print()

    # ── save debug image (Gemini-style) ───────────────────────────────────────
    print(f"Saving debug image → {debug_path}")
    front_raw   = obs[f"{CAMERA}_image"][::-1]          # flip to PIL convention
    debug_img   = draw_red_grid_on_array(front_raw)
    debug_img   = draw_gripper_gizmo(debug_img, obs, env, CAMERA)
    debug_img   = draw_last_target(debug_img, last_u, last_v)
    if args.delta_u != 0.0 or args.delta_v != 0.0:
        debug_img = draw_corrected_target(debug_img, last_u, last_v, new_u, new_v)
    Image.fromarray(debug_img).save(debug_path)

    # ── motion control (mirrors baseline.py exactly) ──────────────────────────
    POS_GAIN      = 3.4
    MAX_CART_ACTION = 0.45
    POS_TOL       = 0.005
    HOME_POS      = np.array([0.4, -0.6, 1.4])
    frames        = []

    def record(n=1):
        for _ in range(n):
            frames.append(obs[f"{CAMERA}_image"][::-1])

    def step_towards(target_xyz, gripper_action, steps=10, settle=False):
        nonlocal obs
        target_xyz = np.array(target_xyz, dtype=float)
        for _ in range(steps):
            eef      = obs["robot0_eef_pos"]
            delta    = target_xyz - eef
            if np.linalg.norm(delta) < POS_TOL:
                break
            distance    = np.linalg.norm(delta)
            speed_scale = 1.0 if distance > 0.10 else (0.7 if distance > 0.04 else 0.45)
            action      = np.zeros(7)
            action[:3]  = np.clip(delta * POS_GAIN, -MAX_CART_ACTION * speed_scale, MAX_CART_ACTION * speed_scale)
            action[6]   = gripper_action
            obs, *_ = env.step(action)
            record()
        if settle:
            settle_action     = np.zeros(7)
            settle_action[6]  = gripper_action
            obs, *_ = env.step(settle_action)
            record()

    # 1. Retract to home (same as baseline's CLEAR VIEW retract)
    print("Moving arm: retracting to HOME_POS…")
    step_towards(HOME_POS, gripper_action=-1, steps=12)
    record(5)

    # 2. Hover above target
    hover    = target_world.copy()
    hover[2] = target_world[2] + 0.20
    print("Moving arm: hovering above target…")
    step_towards(hover, gripper_action=-1, steps=10)

    # 3. Lower to grasp position
    print("Moving arm: lowering to grasp…")
    step_towards(target_world, gripper_action=-1, steps=8)

    # 4. Close gripper
    print("Moving arm: closing gripper…")
    step_towards(target_world, gripper_action=1, steps=6, settle=True)

    # 5. Lift straight up to HOME height so the object clears the bin walls
    lift_pos    = target_world.copy()
    lift_pos[2] = HOME_POS[2]   # 1.4 m — same height as HOME_POS
    print("Moving arm: lifting to HOME height…")
    step_towards(lift_pos, gripper_action=1, steps=15)

    record(20)   # hold on final frame
    env.close()

    print(f"Saving video      → {video_path}")
    imageio.mimwrite(video_path, frames, fps=20)

    subprocess.run(["open", debug_path])
    subprocess.run(["open", video_path])
    print("Done — debug image and video opened.")


if __name__ == "__main__":
    main()
