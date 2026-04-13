import robosuite as suite
import numpy as np
import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection
import time
import os
import platform
import sys
import datetime
import re
import hashlib
from pathlib import Path
from PIL import Image, ImageDraw

try:
    from dotenv import load_dotenv

    # Always load repo-root .env (cwd often differs, e.g. IDEs or subprocesses).
    _PROJECT_ROOT = Path(__file__).resolve().parent.parent
    load_dotenv(_PROJECT_ROOT / ".env")
except ImportError:
    pass
from robosuite.utils.camera_utils import get_camera_transform_matrix, get_real_depth_map, transform_from_pixels_to_world, project_points_from_world_to_camera
import robosuite.utils.transform_utils as T

# Default cereal pose when not using BASELINE_RANDOMIZE_CEREAL=1
CEREAL_BOX_YAW_DEG = -45.0
DEFAULT_CEREAL_POS_M = np.array([0.08, -0.25, 0.9], dtype=float)
# Random placement: only X and Y vary per (seed, trial_idx). Z and in-plane yaw match
# main-branch fixed pose (CEREAL_BOX_YAW_DEG) so arm–object yaw alignment stays consistent.
RANDOM_CEREAL_X_RANGE_M = (0.03, 0.13)
RANDOM_CEREAL_Y_RANGE_M = (-0.32, -0.18)
# Legacy (unused): Z and yaw were previously sampled; kept for snapshot compatibility.
RANDOM_CEREAL_Z_RANGE_M = (0.885, 0.915)
RANDOM_CEREAL_YAW_DEG_RANGE = (-80.0, 80.0)
# Legacy: was used in integer stream = seed + trial_idx * M; kept for metrics snapshots.
CEREAL_PLACEMENT_RNG_TRIAL_MULTIPLIER = 100_003
POS_GAIN = 3.4
MAX_CART_ACTION = 0.45
POS_TOL = 0.005
MAX_YAW_ACTION = 0.18
YAW_TOL = np.deg2rad(1.5)
DEFAULT_GRIPPER_YAW_DEG = -45.0
# Side observation pose used before re-detecting when the target is classified as occluded.
OCCLUSION_OBS_POS = np.array([0.22, -0.35, 1.28])
# Slip recovery: extra depth (m) below nominal grasp height z_com + GRASP_Z_OFFSET_FROM_COM_M.
# Recomputed from sim COM each time (not stacked on obj_pos). More negative = lower on the box
# so fingers can close where the object actually is (tune if the box tips or misses).
SLIP_RECOVERY_Z_OFFSET_M = -0.11
# Extra Cartesian steps when lowering to the slip grasp (default first grasp uses 8).
SLIP_RECOVERY_DESCENT_STEPS = 20
# Number of closed-gripper hold steps after descending on a slip-recovery retry.
SLIP_RECOVERY_SETTLE_STEPS = 16
# Height above the grasp target that the robot moves to before descending.
GRASP_HOVER_Z_OFFSET_M = 0.08
# Upward distance commanded after closing the gripper to test / complete the lift.
GRASP_LIFT_Z_OFFSET_M = 0.35
# Grasp Z: use sim root-body COM height + this offset (m). Avoids a fixed table plane
# that disagrees with actual object height (depth projection is used only as fallback).
GRASP_Z_OFFSET_FROM_COM_M = 0.0
# Align gripper +X (world XY) to the target object's body axis from sim GT before OWL-ViT.
# Must match **main**: use "y" so approach yaw follows ``object_body_y_planar_yaw_deg``
# (straddle / edge-parallel grasp). "x" uses ``object_body_x_planar_yaw_deg`` instead (~90° off).
GRASP_APPROACH_ALIGN_TO_OBJECT_AXIS = "y"
GRASP_APPROACH_YAW_OFFSET_DEG = 0.0
# Tip detection (simplified cereal): body up vs world Z, and COM height vs standing pose.
CEREAL_TIP_MAX_UP_DOT = 0.7
CEREAL_TIP_MIN_COM_Z_M = 0.86
# World Z (m): object body must be above this to count as lifted. Must stay *above* standing
# height (~0.89 m cereal) — never use 0 or any value at/below rest pose or everything "lifts".
# 0.92–0.95 = stricter; 0.88–0.90 = looser (more false SUCCESS if set too low).
LIFT_SUCCESS_BODY_Z_MIN_M = 0.92

# version worked
# SLIP_RECOVERY_Z_OFFSET_M = -0.09
# SLIP_RECOVERY_SETTLE_STEPS = 16
# GRASP_HOVER_Z_OFFSET_M = 0.1
# GRASP_LIFT_Z_OFFSET_M = 0.3


def _eef_planar_yaw_rad(obs):
    """World XY heading of gripper +X (same convention as rotate_yaw_in_place)."""
    eef_rot = T.quat2mat(obs["robot0_eef_quat"])
    return np.arctan2(eef_rot[1, 0], eef_rot[0, 0])


def _axis_planar_yaw_rad(R_world_body, col):
    """World XY heading of a body frame column from rotation matrix R."""
    v = R_world_body[:2, int(col)]
    return np.arctan2(v[1], v[0])


def _wrap_angle_rad(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def get_target_object_ground_truth(env, target_obj_name):
    """Root-body pose for the robosuite object whose name matches target_obj_name."""
    for obj in env.objects:
        if target_obj_name.lower() in obj.name.lower():
            body_id = env.sim.model.body_name2id(obj.root_body)
            pos = np.array(env.sim.data.body_xpos[body_id], dtype=float)
            quat = np.array(env.sim.data.body_xquat[body_id], dtype=float)
            R = T.quat2mat(quat)
            yaw_x = float(np.rad2deg(_axis_planar_yaw_rad(R, 0)))
            yaw_y = float(np.rad2deg(_axis_planar_yaw_rad(R, 1)))
            return {
                "object_name": obj.name,
                "root_body": obj.root_body,
                "com_xyz": pos.tolist(),
                "quat_wxyz": quat.tolist(),
                "object_body_x_planar_yaw_deg": yaw_x,
                "object_body_y_planar_yaw_deg": yaw_y,
                # Same as body +X planar yaw: world-Z rotation of the box for an upright object.
                "object_body_z_yaw_deg": yaw_x,
            }
    return None


def snapshot_tunable_constants():
    """Scalar knobs in this file that most affect grasp quality / slip."""
    return {
        "CEREAL_BOX_YAW_DEG": CEREAL_BOX_YAW_DEG,
        "DEFAULT_GRIPPER_YAW_DEG": DEFAULT_GRIPPER_YAW_DEG,
        "POS_GAIN": POS_GAIN,
        "MAX_CART_ACTION": MAX_CART_ACTION,
        "POS_TOL": POS_TOL,
        "MAX_YAW_ACTION": MAX_YAW_ACTION,
        "YAW_TOL_RAD": float(YAW_TOL),
        "SLIP_RECOVERY_Z_OFFSET_M": SLIP_RECOVERY_Z_OFFSET_M,
        "SLIP_RECOVERY_DESCENT_STEPS": SLIP_RECOVERY_DESCENT_STEPS,
        "SLIP_RECOVERY_SETTLE_STEPS": SLIP_RECOVERY_SETTLE_STEPS,
        "GRASP_HOVER_Z_OFFSET_M": GRASP_HOVER_Z_OFFSET_M,
        "GRASP_LIFT_Z_OFFSET_M": GRASP_LIFT_Z_OFFSET_M,
        "GRASP_Z_OFFSET_FROM_COM_M": GRASP_Z_OFFSET_FROM_COM_M,
        "GRASP_APPROACH_ALIGN_TO_OBJECT_AXIS": GRASP_APPROACH_ALIGN_TO_OBJECT_AXIS,
        "GRASP_APPROACH_YAW_OFFSET_DEG": GRASP_APPROACH_YAW_OFFSET_DEG,
        "LIFT_SUCCESS_BODY_Z_MIN_M": LIFT_SUCCESS_BODY_Z_MIN_M,
        "RANDOM_CEREAL_X_RANGE_M": RANDOM_CEREAL_X_RANGE_M,
        "RANDOM_CEREAL_Y_RANGE_M": RANDOM_CEREAL_Y_RANGE_M,
        "RANDOM_CEREAL_Z_RANGE_M": RANDOM_CEREAL_Z_RANGE_M,
        "RANDOM_CEREAL_YAW_DEG_RANGE": RANDOM_CEREAL_YAW_DEG_RANGE,
        "CEREAL_PLACEMENT_RNG_TRIAL_MULTIPLIER": CEREAL_PLACEMENT_RNG_TRIAL_MULTIPLIER,
    }


def make_cereal_placement_rng(seed, trial_idx):
    """
    RNG used for random cereal **XY** when BASELINE_RANDOMIZE_CEREAL=1.

    - If ``seed`` is None: new unpredictable (x, y) each run (not reproducible).
    - If ``seed`` is set: **same** ``(seed, trial_idx)`` always yields the **same** (x, y)
      (eval uses ``seed=42+trial`` so trial 0..4 get five distinct layouts).
    - **Z** and **yaw** are fixed to ``DEFAULT_CEREAL_POS_M[2]`` and ``CEREAL_BOX_YAW_DEG``
      (main-branch pose), so the arm still aligns to the box from GT.

    The stream is derived from a stable hash of ``(seed, trial_idx)``.
    """
    if seed is None:
        return np.random.default_rng()
    # Hash mix avoids any accidental stream collision if callers pass inconsistent
    # seed/trial_idx combinations; distinct pairs get distinct PCG64 seeds.
    payload = f"cereal_placement:v2:{int(seed)}:{int(trial_idx)}".encode()
    digest = hashlib.sha256(payload).digest()
    stream = int.from_bytes(digest[:8], "big")
    if stream == 0:
        stream = 1
    return np.random.default_rng(stream)


def sample_random_cereal_placement(rng):
    """Sample (x, y) in the bin; z and yaw match main fixed pose so box/arm yaw stay aligned."""
    x = float(rng.uniform(*RANDOM_CEREAL_X_RANGE_M))
    y = float(rng.uniform(*RANDOM_CEREAL_Y_RANGE_M))
    z = float(DEFAULT_CEREAL_POS_M[2])
    yaw_deg = float(CEREAL_BOX_YAW_DEG)
    pos = np.array([x, y, z], dtype=float)
    return {"pos": pos, "yaw_deg": yaw_deg}


def apply_cereal_pose(env, pos_xyz, yaw_deg):
    """
    Place the cereal box at pos_xyz with rotation about world Z given by yaw_deg
    (same free-joint convention as the previous fixed simplify pose).

    This calls ``env.sim.step()`` many times, so the whole scene (including the arm)
    advances while any existing ``obs`` from ``env.reset()`` is **stale**. Callers must
    run ``env.step(zeros)`` (or otherwise refresh observations) before using ``obs``.
    """
    cereal_pos = np.asarray(pos_xyz, dtype=float).reshape(3)
    half_yaw_rad = np.deg2rad(float(yaw_deg)) / 2.0
    cereal_quat = np.array([np.cos(half_yaw_rad), 0.0, 0.0, np.sin(half_yaw_rad)])
    cereal_id = env.sim.model.joint_name2id("Cereal_joint0")
    cereal_idx = env.sim.model.jnt_qposadr[cereal_id]
    env.sim.data.qpos[cereal_idx : cereal_idx + 3] = cereal_pos
    env.sim.data.qpos[cereal_idx + 3 : cereal_idx + 7] = cereal_quat
    env.sim.forward()
    for _ in range(50):
        env.sim.step()
    env.sim.data.qpos[cereal_idx : cereal_idx + 3] = cereal_pos
    env.sim.data.qpos[cereal_idx + 3 : cereal_idx + 7] = cereal_quat
    qvel_idx = env.sim.model.jnt_dofadr[cereal_id]
    env.sim.data.qvel[qvel_idx : qvel_idx + 6] = 0.0
    env.sim.forward()


def apply_cereal_simplified_pose(env):
    """Place cereal at the default fixed pose (backward-compatible)."""
    apply_cereal_pose(env, DEFAULT_CEREAL_POS_M, CEREAL_BOX_YAW_DEG)


def is_cereal_box_tipped(env):
    """True if cereal body is not upright enough or has fallen vs standing pose."""
    try:
        for obj in env.objects:
            if "cereal" not in obj.name.lower():
                continue
            bid = env.sim.model.body_name2id(obj.root_body)
            quat = np.array(env.sim.data.body_xquat[bid], dtype=float)
            R = T.quat2mat(quat)
            up_dot = abs(float(np.dot(R[:, 2], np.array([0.0, 0.0, 1.0]))))
            com_z = float(env.sim.data.body_xpos[bid][2])
            if up_dot < CEREAL_TIP_MAX_UP_DOT or com_z < CEREAL_TIP_MIN_COM_Z_M:
                return True
        return False
    except Exception:
        return False


def maybe_reset_cereal_if_tipped(env, target_obj_name, episode_cereal_pose):
    """
    If the cereal box tipped, restore this episode's nominal cereal pose (for retries).
    episode_cereal_pose: dict with keys "pos" (3,), "yaw_deg" (float).
    Returns True if a reset was applied.
    """
    if "cereal" not in (target_obj_name or "").lower():
        return False
    if not is_cereal_box_tipped(env):
        return False
    print(
        "--- Cereal tipped: resetting box to episode nominal pose before next attempt ---"
    )
    apply_cereal_pose(env, episode_cereal_pose["pos"], episode_cereal_pose["yaw_deg"])
    return True


def simplify_environment(env):
    """Hide non-target clutter; cereal placement is done separately via apply_cereal_pose."""
    print("--- SIMPLIFYING ENVIRONMENT: 1-Bin / 1-Object Mode (Cereal only) ---")
    try:
        for obj_name in ["Bread_joint0", "Can_joint0", "Milk_joint0"]:
            j_id = env.sim.model.joint_name2id(obj_name)
            q_idx = env.sim.model.jnt_qposadr[j_id]
            env.sim.data.qpos[q_idx : q_idx + 3] = np.array([5.0, 5.0, -1.0])
        env.sim.forward()
    except Exception as e:
        print(f"Warning: Physics simplification failed: {e}")

def draw_gripper_gizmo(img_array, obs, env, camera_name):
    """Draw RGB axes on the robot gripper to show orientation in the VLM image."""
    try:
        from PIL import ImageDraw
        image = Image.fromarray(img_array)
        draw = ImageDraw.Draw(image)
        h, w = img_array.shape[:2]
        
        # 1. Get EEF state
        eef_pos = obs['robot0_eef_pos']
        eef_quat = obs['robot0_eef_quat']
        eef_rot = T.quat2mat(eef_quat)
        
        # 2. Project EEF center
        cam_transform = get_camera_transform_matrix(env.sim, camera_name, h, w)
        px_center = project_points_from_world_to_camera(eef_pos.reshape(1,3), cam_transform, h, w)[0]
        # v, u flip for PIL (x, y)
        c_y, c_x = px_center[0], px_center[1]
        
        # 3. Project axis endpoints (length 0.1m)
        axis_len = 0.1
        # X-axis (Red): Heading of fingers
        x_end = eef_pos + eef_rot[:, 0] * axis_len
        px_x = project_points_from_world_to_camera(x_end.reshape(1,3), cam_transform, h, w)[0]
        # Y-axis (Green): Grasping axis
        y_end = eef_pos + eef_rot[:, 1] * axis_len
        px_y = project_points_from_world_to_camera(y_end.reshape(1,3), cam_transform, h, w)[0]
        
        # Draw lines
        draw.line([(c_x, c_y), (px_x[1], px_x[0])], fill=(255, 0, 0), width=3) # Red X
        draw.line([(c_x, c_y), (px_y[1], px_y[0])], fill=(0, 255, 0), width=3) # Green Y
        return np.array(image)
    except Exception as e:
        print(f"Gizmo Draw Error: {e}")
        return img_array

def create_frontview_image(obs, env):
    """Render a front-view frame with red pixel-coordinate grid and gripper gizmo."""
    front = obs["frontview_image"][::-1]
    front_grid = draw_red_grid_on_array(front)
    return draw_gripper_gizmo(front_grid, obs, env, "frontview")

def draw_red_grid_on_array(img_array):
    """Draw a 32-pixel red coordinate grid on the image array."""
    try:
        import PIL.Image as Image
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
    except:
        return img_array


def draw_owlvit_box_on_rgb(img_rgb, box_xyxy, score_text=""):
    """
    Draw OWL-ViT detection box on a frontview RGB frame (same pixel space as model input).
    img_rgb: (H, W, 3) uint8; box_xyxy: [x0, y0, x1, y1] in pixels.
    """
    try:
        from PIL import ImageDraw

        im = Image.fromarray(np.asarray(img_rgb, dtype=np.uint8).copy())
        draw = ImageDraw.Draw(im)
        x0, y0, x1, y1 = [float(x) for x in np.asarray(box_xyxy).ravel()[:4]]
        draw.rectangle([x0, y0, x1, y1], outline=(0, 255, 0), width=4)
        cap = score_text or "OWL-ViT"
        ty = max(0, int(y0) - 18)
        draw.text((int(x0), ty), cap, fill=(0, 255, 0))
        return np.array(im)
    except Exception as e:
        print(f"OWL overlay draw error: {e}")
        return np.asarray(img_rgb)


# Hold OWL detection overlay in the MP4 so the box is visible before the grasp motion (~1s at 20fps).
OWL_DETECTION_VIDEO_HOLD_FRAMES = 20


def get_max_attempts_for_condition(condition):
    if condition == "feedback":
        return 2
    if condition == "feedback_double":
        return 3
    match = re.fullmatch(r"feedback_(\d+)", condition or "")
    if match:
        return max(1, int(match.group(1)))
    return 1

def failure_type_implies_grasp_success(failure_type):
    return failure_type == "slip_after_contact"


def instruction_to_target_object_name(instruction):
    """Map free-text instruction to robosuite object name, or None if unknown."""
    if not instruction:
        return None
    s = instruction.lower()
    if "milk" in s:
        return "Milk"
    if "bread" in s:
        return "Bread"
    if "cereal" in s:
        return "Cereal"
    if "can" in s:
        return "Can"
    return None


def run_baseline(instruction="pick the milk", condition="feedback", trial_idx=0, seed=None, processor=None, model=None, device=None):
    # Normalize so "Feedback", " feedback ", etc. still enable recovery; must match get_max_attempts_for_condition.
    if condition is not None:
        condition = str(condition).strip().lower()

    placement_only = os.environ.get("BASELINE_CEREAL_PLACEMENT_ONLY", "0").strip() == "1"

    print(f"\n--- Starting Trial {trial_idx} | Condition: {condition} ---")
    print(f"Language Instruction: '{instruction}'")
    
    start_time = time.time()
    metrics = {
        "task_success": False,
        "wrong_object": False,
        "grasp_success": False,
        "recovery_success": False,
        "attempts": 1,
        "latency": 0.0,
        "failure_type": "",
        "failed_checkpoint": "",
        "explanation": ""
    }
    max_attempts = get_max_attempts_for_condition(condition)
    print(
        f"Max grasp attempts (initial + retries) for this condition: {max_attempts}. "
        f"Use condition='feedback' or 'feedback_3' for retries; 'feedback_1' or 'baseline' allows only the first grasp."
    )

    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    # 1. Perception Step (OWL-ViT initialization)
    if device is None:
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    if not placement_only and (processor is None or model is None):
        print("Loading OWL-ViT model onto device...")
        processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
        model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32").to(device)

    # 2. Environment Setup
    print("Initializing Robosuite 'PickPlace' environment with Panda arm...")
    IMG_H, IMG_W = 512, 512

    # Always create a timestamped run directory for artifacts and video.
    # Set BASELINE_RENDER=0 to disable video recording (frames + logs still saved).
    import imageio
    timestamp = datetime.datetime.now().strftime("%m%d_%I_%M_%S_%p").lower()
    run_name = f"run_{condition}_trial_{trial_idx}_{timestamp}"
    run_dir = os.path.join("runs", run_name)
    os.makedirs(run_dir, exist_ok=True)
    print(f"Run artifacts → {run_dir}")
    tuning_grasp_records = []
    initial_detection_vs_gt = None

    def get_attempt_dir(attempt_num):
        attempt_dir = os.path.join(run_dir, f"attempt_{attempt_num}")
        os.makedirs(attempt_dir, exist_ok=True)
        return attempt_dir

    video_enabled = os.environ.get("BASELINE_RENDER", "1") != "0"
    video_writer = None
    current_vid_path = os.path.join(run_dir, "attempt_1.mp4")
    if video_enabled:
        video_writer = imageio.get_writer(current_vid_path, fps=20)

    # The default PickPlaceSingle loads a Single Object, PickPlace loads 4 (Milk, Bread, Cereal, Can)
    # We will use PickPlace to have clutter.
    env = suite.make(
        env_name="PickPlace", 
        robots="Panda",             
        controller_configs=suite.load_composite_controller_config(controller="BASIC"),
        has_renderer=False,
        use_camera_obs=True,      # Enabled for OWL-ViT vision
        camera_names=["frontview", "birdview"],
        camera_heights=IMG_H,
        camera_widths=IMG_W,
        camera_depths=True,
        has_offscreen_renderer=True, # Need offscreen to render camera obs
        control_freq=20,
        render_camera="frontview",
    )


    def step_towards(current_obs, target_xyz, gripper_action, steps=10, settle=False):
        """
        Move EEF toward target using cosine-eased waypoints along the segment from the
        pose at call time to the goal. This matches manual_grasp.py and avoids taking
        zero env.step() calls when stale obs or tolerance would skip the whole segment.
        """
        target_xyz = np.array(target_xyz, dtype=float)
        start_xyz = np.asarray(current_obs["robot0_eef_pos"], dtype=float).copy()
        n = max(int(steps), 1)

        for step_idx in range(n):
            progress = (step_idx + 1) / float(n)
            eased = 0.5 - 0.5 * np.cos(np.pi * progress)
            waypoint = start_xyz + (target_xyz - start_xyz) * eased
            current_eef = current_obs["robot0_eef_pos"]
            delta = waypoint - current_eef
            if np.linalg.norm(target_xyz - current_eef) < POS_TOL:
                break

            action = np.zeros(7)
            distance = float(np.linalg.norm(target_xyz - current_eef))
            speed_scale = 1.0 if distance > 0.10 else (0.7 if distance > 0.04 else 0.45)
            action[:3] = np.clip(
                delta * POS_GAIN,
                -MAX_CART_ACTION * speed_scale,
                MAX_CART_ACTION * speed_scale,
            )
            action[6] = gripper_action
            current_obs, reward, done, info = env.step(action)
            if video_enabled and video_writer:
                video_writer.append_data(current_obs["frontview_image"][::-1])

        if settle:
            settle_action = np.zeros(7)
            settle_action[6] = gripper_action
            current_obs, reward, done, info = env.step(settle_action)
            if video_enabled and video_writer:
                video_writer.append_data(current_obs["frontview_image"][::-1])
        return current_obs

    def rotate_yaw_in_place(current_obs, yaw_delta_rad, gripper_action, steps=20):
        def wrap_to_pi(angle_rad):
            return (angle_rad + np.pi) % (2.0 * np.pi) - np.pi

        def current_eef_yaw(obs_ref):
            eef_rot = T.quat2mat(obs_ref["robot0_eef_quat"])
            return np.arctan2(eef_rot[1, 0], eef_rot[0, 0])

        normalized_delta = wrap_to_pi(yaw_delta_rad)
        if abs(normalized_delta) < YAW_TOL:
            return current_obs

        start_yaw = current_eef_yaw(current_obs)
        target_yaw = wrap_to_pi(start_yaw + normalized_delta)
        for _ in range(steps):
            yaw_error = wrap_to_pi(target_yaw - current_eef_yaw(current_obs))
            if abs(yaw_error) < YAW_TOL:
                break

            action = np.zeros(7)
            action[5] = np.clip(yaw_error * 0.8, -MAX_YAW_ACTION, MAX_YAW_ACTION)
            action[6] = gripper_action
            current_obs, _, _, _ = env.step(action)
            if video_enabled and video_writer:
                video_writer.append_data(current_obs["frontview_image"][::-1])

        return current_obs

    def rotate_yaw_to_world_yaw(current_obs, target_yaw_rad, gripper_action, steps=40):
        """Drive gripper planar yaw to an absolute world-frame heading (gripper +X in XY)."""

        def wrap_to_pi(angle_rad):
            return (angle_rad + np.pi) % (2.0 * np.pi) - np.pi

        def current_eef_yaw(obs_ref):
            eef_rot = T.quat2mat(obs_ref["robot0_eef_quat"])
            return np.arctan2(eef_rot[1, 0], eef_rot[0, 0])

        target_yaw = wrap_to_pi(target_yaw_rad)
        for _ in range(steps):
            yaw_error = wrap_to_pi(target_yaw - current_eef_yaw(current_obs))
            if abs(yaw_error) < YAW_TOL:
                break

            action = np.zeros(7)
            action[5] = np.clip(yaw_error * 0.8, -MAX_YAW_ACTION, MAX_YAW_ACTION)
            action[6] = gripper_action
            current_obs, _, _, _ = env.step(action)
            if video_enabled and video_writer:
                video_writer.append_data(current_obs["frontview_image"][::-1])

        return current_obs

    def score_detection_candidate(box, raw_score, camera_name, previous_target_pos=None):
        center_u = (box[0] + box[2]) / 2.0
        center_v = (box[1] + box[3]) / 2.0
        width = max(1.0, box[2] - box[0])
        height = max(1.0, box[3] - box[1])
        area_term = min((width * height) / float(IMG_H * IMG_W), 1.0)
        score = float(raw_score) + 0.1 * area_term
        if previous_target_pos is not None:
            prev_px = project_points_from_world_to_camera(
                previous_target_pos.reshape(1, 3),
                get_camera_transform_matrix(env.sim, camera_name, IMG_H, IMG_W),
                IMG_H,
                IMG_W,
            )[0]
            pixel_dist = np.linalg.norm(np.array([center_v, center_u]) - prev_px)
            score -= 0.0015 * float(pixel_dist)
        return score

    def detect_target_pos(current_obs, query, camera_name="frontview", previous_target_pos=None, top_k=3):
        img_key = f"{camera_name}_image"
        depth_key = f"{camera_name}_depth"
        img_raw = current_obs[img_key][::-1]
        texts_r = [[f"a photo of a {query}"]]
        inp_r = processor(text=texts_r, images=img_raw, return_tensors="pt").to(device)
        with torch.no_grad():
            out_r = model(**inp_r)
        ts_r = torch.tensor([img_raw.shape[:2]]).to(device)
        res_r = processor.post_process_grounded_object_detection(
            outputs=out_r, target_sizes=ts_r, text_labels=texts_r, threshold=0.0
        )
        if len(res_r[0]["scores"]) == 0:
            return None

        boxes = res_r[0]["boxes"].cpu().numpy()
        scores = res_r[0]["scores"].cpu().numpy()
        ranked = sorted(
            [
                (
                    score_detection_candidate(
                        box,
                        score,
                        camera_name,
                        previous_target_pos=previous_target_pos,
                    ),
                    float(score),
                    box,
                )
                for box, score in zip(boxes, scores)
            ],
            key=lambda item: item[0],
            reverse=True,
        )[:top_k]

        best_ranked_score, best_raw_score, best_box = ranked[0]
        best_u = (best_box[0] + best_box[2]) / 2.0
        best_v = (best_box[1] + best_box[3]) / 2.0
        real_depth_local = get_real_depth_map(env.sim, current_obs[depth_key])
        cam_mat_local = np.linalg.inv(get_camera_transform_matrix(env.sim, camera_name, IMG_H, IMG_W))
        best_pos = transform_from_pixels_to_world(
            np.array([(IMG_H - 1) - best_v, best_u]),
            real_depth_local,
            cam_mat_local,
        )
        best_pos[2] = initial_owlvit_3d[2]
        print(
            f"  Re-detection ({camera_name}): chose score={best_raw_score:.3f}, "
            f"ranked={best_ranked_score:.3f}, center=({best_u:.1f}, {best_v:.1f})"
        )
        return best_pos

    # 3. Task Execution
    obs = env.reset()
    
    # SIMPLIFY: hide clutter; then place cereal (fixed or random per env / seed).
    simplify_environment(env)
    randomize_cereal = os.environ.get("BASELINE_RANDOMIZE_CEREAL", "0").strip() == "1"
    if randomize_cereal:
        rng = make_cereal_placement_rng(seed, trial_idx)
        episode_cereal_pose = sample_random_cereal_placement(rng)
        print(
            "--- RANDOM CEREAL PLACEMENT (XY by seed/trial; Z & yaw = main fixed pose): "
            f"trial_idx={trial_idx}, seed={seed!r}, "
            f"pos={np.round(episode_cereal_pose['pos'], 4)} m, "
            f"yaw_deg={episode_cereal_pose['yaw_deg']:.2f} "
            "(OWL overlay unchanged; arm aligns to box GT yaw) ---"
        )
    else:
        episode_cereal_pose = {
            "pos": DEFAULT_CEREAL_POS_M.copy(),
            "yaw_deg": float(CEREAL_BOX_YAW_DEG),
        }
        print(
            "--- FIXED CEREAL PLACEMENT: "
            f"pos={np.round(episode_cereal_pose['pos'], 4)} m, "
            f"yaw_deg={episode_cereal_pose['yaw_deg']:.2f} "
            "(set BASELINE_RANDOMIZE_CEREAL=1 for random pose per run) ---"
        )
    apply_cereal_pose(env, episode_cereal_pose["pos"], episode_cereal_pose["yaw_deg"])
    # apply_cereal_pose() runs many sim.substeps; obs from reset() no longer matches sim.
    hold = np.zeros(int(env.action_dim))
    obs, _, _, _ = env.step(hold)

    if placement_only:
        print(
            "--- BASELINE_CEREAL_PLACEMENT_ONLY=1: skipping arm motion, OWL-ViT, grasp, and Gemini ---"
            "(unset BASELINE_CEREAL_PLACEMENT_ONLY for full video with OWL-ViT + pick attempt; "
            "use BASELINE_SKIP_GEMINI=1 if you only want to skip Gemini on failure.) ---"
        )
        metrics["placement_only"] = True
        metrics["failure_type"] = "placement_only_skip"
        target_obj_name = instruction_to_target_object_name(instruction)
        try:
            png_path = os.path.join(run_dir, "cereal_placement_view.png")
            Image.fromarray(create_frontview_image(obs, env)).save(png_path)
            print(f"Placement snapshot → {png_path}")
        except Exception:
            pass
        if video_enabled and video_writer:
            try:
                view_rgb = obs["frontview_image"][::-1]
                # Many decoders reject 0-frame MP4s; duplicate one frame for a short valid clip.
                for _ in range(20):
                    video_writer.append_data(view_rgb)
                video_writer.close()
                print(f"Video saved → {current_vid_path}")
            except Exception as e:
                print(f"Warning: could not write placement-only video: {e}")
        elif not video_enabled:
            print(
                "No video (BASELINE_RENDER=0); use cereal_placement_view.png or unset BASELINE_RENDER."
            )
        env.close()
        metrics["latency"] = time.time() - start_time
        try:
            import json as _json

            with open(os.path.join(run_dir, "trial_summary.json"), "w") as f:
                _json.dump(
                    {
                        "condition": condition,
                        "instruction": instruction,
                        "trial_idx": trial_idx,
                        "run_dir": run_dir,
                        "target_object": target_obj_name,
                        "cereal_episode_pose_m": {
                            "pos": np.asarray(episode_cereal_pose["pos"], dtype=float).tolist(),
                            "yaw_deg": float(episode_cereal_pose["yaw_deg"]),
                        },
                        "baseline_randomize_cereal": bool(randomize_cereal),
                        **metrics,
                    },
                    f,
                    indent=2,
                )
            print(f"Trial summary → {os.path.join(run_dir, 'trial_summary.json')}")
        except Exception:
            pass
        return metrics

    # CLEAR VIEW: retract along the same smoothed motion profile used for grasping.
    print("--- CLEAR VIEW: Retracting arm for initial perception ---")
    retract_pos = np.array([0.4, -0.6, 1.4])
    obs = step_towards(obs, retract_pos, gripper_action=-1, steps=28)

    target_obj_name = instruction_to_target_object_name(instruction)

    if not target_obj_name:
        print("Failure Reasoning: 'wrong-object selection'")
        print("Explanation: The perception model could not map the instruction to an object.")
        return False

    approach_yaw_world_deg_used = float(DEFAULT_GRIPPER_YAW_DEG)
    approach_axis_used = "fallback"
    gt_pose_for_yaw = get_target_object_ground_truth(env, target_obj_name)
    if gt_pose_for_yaw is not None:
        ax = str(GRASP_APPROACH_ALIGN_TO_OBJECT_AXIS).lower()
        if ax == "x":
            base_deg = float(gt_pose_for_yaw["object_body_x_planar_yaw_deg"])
        else:
            base_deg = float(gt_pose_for_yaw["object_body_y_planar_yaw_deg"])
        approach_yaw_deg = base_deg + GRASP_APPROACH_YAW_OFFSET_DEG
        approach_yaw_world_deg_used = float(approach_yaw_deg)
        approach_axis_used = ax
        print(
            f"--- APPROACH YAW: align gripper +X to object body {ax}-axis in world XY "
            f"(box rotation → world yaw {approach_yaw_deg:.1f}°) before OWL-ViT / grasp ---"
        )
        obs = rotate_yaw_to_world_yaw(
            obs, np.deg2rad(approach_yaw_deg), gripper_action=-1, steps=40
        )
    else:
        print(
            f"--- APPROACH YAW: no sim GT for '{target_obj_name}'; "
            f"using fallback world yaw {DEFAULT_GRIPPER_YAW_DEG:.0f}° ---"
        )
        obs = rotate_yaw_to_world_yaw(
            obs, np.deg2rad(DEFAULT_GRIPPER_YAW_DEG), gripper_action=-1, steps=40
        )

    owlvit_frontview = create_frontview_image(obs, env)
    try:
        Image.fromarray(owlvit_frontview).save(os.path.join(run_dir, "owlvit_clear_view.png"))
    except Exception:
        pass
    
    if video_enabled and video_writer:
        video_writer.append_data(obs["frontview_image"][::-1])

    print(f"Perception matched object: {target_obj_name}")

    # Process camera obs and detect with OWL-ViT
    img = obs["frontview_image"][::-1] # robosuite is upside down vertically
    depth = obs["frontview_depth"]
    
    prompts = {
        "Milk": "milk carton",
        "Bread": "loaf of bread",
        "Cereal": "cereal box",
        "Can": "soda can"
    }
    texts = [[f"a photo of a {prompts.get(target_obj_name, target_obj_name.lower())}"]]
    inputs = processor(text=texts, images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    
    target_sizes = torch.tensor([img.shape[:2]]).to(device)
    # Deprecation warning fix: using post_process_grounded_object_detection
    results = processor.post_process_grounded_object_detection(outputs=outputs, target_sizes=target_sizes, text_labels=texts, threshold=0.0)
    
    if len(results[0]["scores"]) == 0:
        print("Failure Reasoning: 'target occluded' / not found")
        print("Explanation: OWL-ViT could not visually detect the object in the camera view.")
        if video_enabled and video_writer: video_writer.close()
        return False
        
    # Get the highest scoring box
    best_idx = torch.argmax(results[0]["scores"])
    box = results[0]["boxes"][best_idx].cpu().numpy()
    score = results[0]["scores"][best_idx].cpu().numpy()
    
    print(f"OWL-ViT Detected '{target_obj_name}' with score {score:.3f}")

    score_f = float(np.asarray(score).reshape(-1)[0])
    owlvit_overlay = draw_owlvit_box_on_rgb(
        img, box, score_text=f"OWL-ViT {score_f:.2f}"
    )
    try:
        Image.fromarray(owlvit_overlay).save(
            os.path.join(run_dir, "owlvit_detection_overlay.png")
        )
    except Exception:
        pass
    if video_enabled and video_writer:
        for _ in range(OWL_DETECTION_VIDEO_HOLD_FRAMES):
            video_writer.append_data(owlvit_overlay)

    # Calculate pixel center
    u = (box[0] + box[2]) / 2.0
    v = (box[1] + box[3]) / 2.0
    
    # Original image coordinate matching for depth map
    orig_v = (IMG_H - 1) - v
    
    # Project 2D to 3D using real depth map
    real_depth = get_real_depth_map(env.sim, depth)
    cam_mat = np.linalg.inv(get_camera_transform_matrix(env.sim, "frontview", IMG_H, IMG_W))
    
    # transform_from_pixels_to_world inherently expects [row, col] format (i.e. [v, u])
    # so that bilinear interpolation correctly maps im[v, u] and cam_pts traces u*z, v*z.
    pixels = np.array([orig_v, u])
    
    pt_3d = transform_from_pixels_to_world(pixels, real_depth, cam_mat)
    z_projected = float(pt_3d[2])
    gt_pose = get_target_object_ground_truth(env, target_obj_name)
    if gt_pose is not None:
        z_com = float(np.asarray(gt_pose["com_xyz"], dtype=float)[2])
        pt_3d[2] = z_com + GRASP_Z_OFFSET_FROM_COM_M
        print(
            f"Grasp Z from sim COM ({z_com:.4f} m) + offset {GRASP_Z_OFFSET_FROM_COM_M:.4f} m "
            f"(projected depth Z was {z_projected:.4f} m)."
        )
    else:
        pt_3d[2] = z_projected
        print(
            f"Grasp Z from depth projection ({z_projected:.4f} m); no matching object in sim for COM."
        )

    initial_owlvit_3d = pt_3d.copy()

    print(f"Targeting '{target_obj_name}' at 3D coordinate {pt_3d}...")

    # VLM Target 3D Position
    obj_pos = pt_3d.copy()
    # OWL overlay loop appended duplicate pixels without stepping physics; refresh obs once.
    obs, _, _, _ = env.step(np.zeros(int(env.action_dim)))

    try:
        gt0 = get_target_object_ground_truth(env, target_obj_name)
        if gt0:
            cam_tf = get_camera_transform_matrix(env.sim, "frontview", IMG_H, IMG_W)
            gt_com = np.array(gt0["com_xyz"], dtype=float)
            pix = project_points_from_world_to_camera(
                gt_com.reshape(1, 3), cam_tf, IMG_H, IMG_W
            )[0]
            gt_v, gt_u = float(pix[0]), float(pix[1])
            initial_detection_vs_gt = {
                "owlvit_box_center_uv": [float(u), float(v)],
                "gt_com_projected_to_frontview_vu": [gt_v, gt_u],
                "pixel_delta_detection_minus_gt_vu": [float(v) - gt_v, float(u) - gt_u],
                "commanded_initial_target_xyz_m": pt_3d.tolist(),
                "gt_com_xyz_m": gt0["com_xyz"],
                "delta_commanded_minus_gt_com_m": (pt_3d - gt_com).tolist(),
                "gt_body_axes_planar_yaw_deg": {
                    "x": gt0["object_body_x_planar_yaw_deg"],
                    "y": gt0["object_body_y_planar_yaw_deg"],
                    "z_box_yaw_deg": gt0.get("object_body_z_yaw_deg"),
                },
                "approach_yaw_world_deg": float(approach_yaw_world_deg_used),
                "approach_axis_aligned_to": approach_axis_used,
                "fallback_yaw_constant_deg": float(DEFAULT_GRIPPER_YAW_DEG),
            }
    except Exception as e:
        initial_detection_vs_gt = {"error": str(e)}

    HOME_POS = np.array([0.4, -0.6, 1.4])

    # Check outcome
    # Check outcome helper
    def check_objects_lifted(env_ref, target_name):
        lifted_any = False
        target_picked = False
        wrong_picked = False
        for o in env_ref.objects:
            body_id = env_ref.sim.model.body_name2id(o.root_body)
            z_pos = env_ref.sim.data.body_xpos[body_id][2]
            if z_pos > LIFT_SUCCESS_BODY_Z_MIN_M:
                lifted_any = True
                if target_name.lower() in o.name.lower():
                    target_picked = True
                else:
                    wrong_picked = True
        return lifted_any, target_picked, wrong_picked

    # Temporal evidence frames captured at each grasp checkpoint for Stage 1 classification
    frames = {}

    def perform_grasp_attempt(
        current_obs,
        target_pos,
        settle_steps=6,
        pre_yaw_deg=0.0,
        descent_steps=24,
    ):
        attempt_frames = {}

        print("Action Plan: Hovering above object...")
        hover_pos = target_pos.copy()
        hover_pos[2] += GRASP_HOVER_Z_OFFSET_M
        current_obs = step_towards(current_obs, hover_pos, gripper_action=-1, steps=32)
        attempt_frames["pre_hover"] = create_frontview_image(current_obs, env)

        # Approach yaw is applied once before OWL-ViT (rotate_yaw_to_world_yaw); main does not
        # rotate again after hover—only optional recovery pre_yaw_deg.

        if abs(pre_yaw_deg) > 0.5:
            print(f"Action Plan: Rotating gripper {pre_yaw_deg:.0f}° (recovery) before descent...")
            current_obs = rotate_yaw_in_place(
                current_obs, np.deg2rad(pre_yaw_deg), gripper_action=-1, steps=20
            )

        print(f"Action Plan: Lowering to grasp ({descent_steps} steps)...")
        grasp_pos = target_pos.copy()
        current_obs = step_towards(
            current_obs, grasp_pos, gripper_action=-1, steps=descent_steps
        )
        attempt_frames["contact"] = create_frontview_image(current_obs, env)
        obs_at_contact = current_obs

        print("Action Plan: Closing gripper...")
        current_obs = step_towards(current_obs, grasp_pos, gripper_action=1, steps=settle_steps, settle=True)
        attempt_frames["post_close"] = create_frontview_image(current_obs, env)

        print("Action Plan: Lifting...")
        lift_pos = grasp_pos.copy()
        lift_pos[2] += GRASP_LIFT_Z_OFFSET_M
        current_obs = step_towards(current_obs, lift_pos, gripper_action=1, steps=28)
        attempt_frames["post_lift"] = create_frontview_image(current_obs, env)

        lifted_any, target_ok, wrong_ok = check_objects_lifted(env, target_obj_name)

        try:
            gt = get_target_object_ground_truth(env, target_obj_name)
            yaw_e = _eef_planar_yaw_rad(obs_at_contact)
            rec = {
                "commanded_target_xyz_m": grasp_pos.tolist(),
                "hover_xyz_m": hover_pos.tolist(),
                "pre_yaw_deg": float(pre_yaw_deg),
                "descent_steps": int(descent_steps),
                "settle_steps": int(settle_steps),
                "lift_delta_z_m": float(GRASP_LIFT_Z_OFFSET_M),
                "at_contact_eef_xyz_m": obs_at_contact["robot0_eef_pos"].tolist(),
                "at_contact_eef_planar_yaw_deg": float(np.rad2deg(yaw_e)),
                "lifted_outcome": {
                    "lifted_any": bool(lifted_any),
                    "target_picked": bool(target_ok),
                    "wrong_picked": bool(wrong_ok),
                },
            }
            if gt:
                gcom = np.array(gt["com_xyz"], dtype=float)
                yaw_x = np.deg2rad(gt["object_body_x_planar_yaw_deg"])
                yaw_y = np.deg2rad(gt["object_body_y_planar_yaw_deg"])
                rec["ground_truth_com_xyz_m"] = gt["com_xyz"]
                rec["ground_truth_body_axes_planar_yaw_deg"] = {
                    "x": gt["object_body_x_planar_yaw_deg"],
                    "y": gt["object_body_y_planar_yaw_deg"],
                }
                rec["errors_m"] = {
                    "commanded_grasp_minus_gt_com": (grasp_pos - gcom).tolist(),
                    "eef_contact_minus_gt_com": (
                        obs_at_contact["robot0_eef_pos"] - gcom
                    ).tolist(),
                }
                rec["errors_deg"] = {
                    "eef_planar_yaw_minus_object_body_x_axis": float(
                        np.rad2deg(_wrap_angle_rad(yaw_e - yaw_x))
                    ),
                    "eef_planar_yaw_minus_object_body_y_axis": float(
                        np.rad2deg(_wrap_angle_rad(yaw_e - yaw_y))
                    ),
                }
            tuning_grasp_records.append(rec)
        except Exception as e:
            tuning_grasp_records.append({"error": str(e)})

        return current_obs, attempt_frames, lifted_any, target_ok, wrong_ok

    obs, frames, lifted_any, target_picked, wrong_picked = perform_grasp_attempt(obs, obj_pos)

    if target_picked:
        print("\nOutcome: SUCCESS")
        print("The baseline policy successfully completed the language instruction.")
        metrics["task_success"] = True
        metrics["grasp_success"] = True
    else:
        print("\nOutcome: FAILURE")
        if lifted_any:
            metrics["wrong_object"] = True
            metrics["grasp_success"] = True

        skip_gemini = os.environ.get("BASELINE_SKIP_GEMINI", "0").strip() == "1"
        feedback_enabled = (
            condition == "explanation_only" or condition.startswith("feedback")
        ) and not skip_gemini
        if feedback_enabled:
            from src.explanation_module import classify_failure, OPENROUTER_MODEL
            import json as _json

            OWLVIT_PROMPTS = {
                "Milk":   ("milk carton",    "white milk carton with red label"),
                "Bread":  ("loaf of bread",  "tan rectangular bread loaf box"),
                "Cereal": ("cereal box",     "red rectangular cereal box"),
                "Can":    ("soda can",       "red cylindrical metal soda can"),
            }

            def redetect_target_pos(current_obs, failure_type, disambiguation=False, previous_target_pos=None):
                """Re-run OWL-ViT from one or more observation views without VLM pixel regression."""
                base_p, disam_p = OWLVIT_PROMPTS.get(
                    target_obj_name, (target_obj_name.lower(), target_obj_name.lower())
                )
                query = disam_p if disambiguation else base_p
                if failure_type == "target_occluded":
                    print("  Occlusion recovery: shifting to a side observation pose and checking birdview first.")
                    current_obs = step_towards(current_obs, OCCLUSION_OBS_POS, gripper_action=-1, steps=12)
                    for camera_name in ("birdview", "frontview"):
                        new_pos = detect_target_pos(
                            current_obs,
                            query,
                            camera_name=camera_name,
                            previous_target_pos=previous_target_pos,
                        )
                        if new_pos is not None:
                            return current_obs, new_pos
                else:
                    new_pos = detect_target_pos(
                        current_obs,
                        query,
                        camera_name="frontview",
                        previous_target_pos=previous_target_pos,
                    )
                    if new_pos is not None:
                        return current_obs, new_pos

                print("  Re-detection: no object found, falling back to initial OWL-ViT target.")
                return current_obs, initial_owlvit_3d.copy()

            while not metrics["task_success"]:
                # --- Reset: retract arm so scene is unoccluded for the final frame ---
                print("--- FULL RETRACT: Arm to home for clean scene capture ---")
                obs = step_towards(obs, HOME_POS, gripper_action=-1, steps=10)
                frames["retracted"] = create_frontview_image(obs, env)

                # Save every frame that is being sent to Gemini for this attempt, including
                # any auxiliary birdview frames added for debugging or occlusion handling.
                FRAME_ORDER = ["pre_hover", "contact", "post_close", "post_lift", "retracted"]
                ordered_frame_names = FRAME_ORDER + sorted(
                    key for key in frames.keys() if key not in FRAME_ORDER
                )
                attempt_num = metrics["attempts"]
                attempt_dir = get_attempt_dir(attempt_num)
                frames_dir = os.path.join(attempt_dir, "gemini_frames")
                os.makedirs(frames_dir, exist_ok=True)
                for i, fname in enumerate(ordered_frame_names):
                    if fname in frames:
                        try:
                            Image.fromarray(frames[fname]).save(
                                os.path.join(frames_dir, f"frame_{i+1:02d}_{fname}.png")
                            )
                        except Exception:
                            pass

                # --- Stage 1: Classify failure from 5 temporal frames ---
                print(f"Querying {OPENROUTER_MODEL} for failure classification (5 frames)...")
                try:
                    failure_result = classify_failure(target_obj_name, frames)
                except Exception as e:
                    print(f"Failure classification error: {e}")
                    failure_result = None

                if not failure_result:
                    print("Gemini returned no classification. Ending trial.")
                    break

                failure_type = failure_result["failure_type"]
                failed_checkpoint = failure_result.get("failed_checkpoint", "unknown")
                print(f"\n[GEMINI FAILURE CLASSIFICATION]")
                print(f"  Failed checkpoint : {failed_checkpoint}")
                print(f"  Failure type      : {failure_type}")
                print(f"  Explanation       : {failure_result['explanation']}")
                print(f"  Confidence        : {failure_result['confidence']:.2f}")

                metrics["failure_type"] = failure_type
                metrics["failed_checkpoint"] = failed_checkpoint
                metrics["explanation"] = failure_result["explanation"]
                if failure_type_implies_grasp_success(failure_type):
                    metrics["grasp_success"] = True

                prompt_text = failure_result.get("prompt_text", "")
                prompt_transcript = failure_result.get("prompt_transcript", "")
                try:
                    with open(os.path.join(attempt_dir, "gemini_prompt.txt"), "w") as f:
                        f.write(prompt_text)
                    if prompt_transcript:
                        with open(
                            os.path.join(attempt_dir, "gemini_user_message_transcript.txt"),
                            "w",
                        ) as f:
                            f.write(prompt_transcript)
                    log = {
                        k: v
                        for k, v in failure_result.items()
                        if k not in ("prompt_text", "prompt_transcript")
                    }
                    with open(os.path.join(attempt_dir, "failure_classification.json"), "w") as f:
                        _json.dump(log, f, indent=2)
                except Exception:
                    pass

                if condition == "explanation_only":
                    break

                if metrics["attempts"] >= max_attempts:
                    print(f"Reached max attempt budget ({max_attempts}). Ending trial.")
                    break

                metrics["attempts"] += 1
                # Duplicate the classification prompt into the next attempt folder so
                # e.g. attempt_1/ and attempt_2/ both carry the same text for review
                # (attempt_2 also gets gemini_prompt.txt if that grasp fails later).
                try:
                    next_attempt_dir = get_attempt_dir(metrics["attempts"])
                    with open(
                        os.path.join(next_attempt_dir, "gemini_prompt_from_previous_attempt.txt"),
                        "w",
                    ) as f:
                        f.write(prompt_text)
                    if prompt_transcript:
                        with open(
                            os.path.join(
                                next_attempt_dir,
                                "gemini_user_message_transcript_from_previous_attempt.txt",
                            ),
                            "w",
                        ) as f:
                            f.write(prompt_transcript)
                except Exception:
                    pass

                if maybe_reset_cereal_if_tipped(
                    env, target_obj_name, episode_cereal_pose
                ):
                    obj_pos = initial_owlvit_3d.copy()

                print(f"\n--- RECOVERY ATTEMPT {metrics['attempts']} | policy={failure_type} ---")

                if video_enabled and video_writer is not None:
                    try:
                        video_writer.close()
                    except Exception:
                        pass
                    current_vid_path = os.path.join(run_dir, f"attempt_{metrics['attempts']}.mp4")
                    try:
                        video_writer = imageio.get_writer(current_vid_path, fps=20)
                    except Exception:
                        pass

                recovery_log = {
                    "failure_type": failure_type,
                    "failed_checkpoint": failed_checkpoint,
                    "policy": None,
                    "parameters": {},
                    "outcome": None,
                }
                target_pos = None

                if failure_type in ("wrong_object", "no_object_reached", "target_occluded"):
                    disambiguation = failure_type == "wrong_object"
                    obs, target_pos = redetect_target_pos(
                        obs,
                        failure_type,
                        disambiguation=disambiguation,
                        previous_target_pos=obj_pos.copy(),
                    )
                    recovery_log["policy"] = "redetect_owlvit"
                    recovery_log["parameters"] = {
                        "disambiguation": disambiguation,
                        "target_pos": target_pos.tolist(),
                    }
                    obs, frames, lifted_any, target_ok, wrong_ok = perform_grasp_attempt(obs, target_pos)

                elif failure_type in ("grasp_pose_bad", "needs_yaw_adjustment"):
                    yaw = 90.0 if failure_type == "needs_yaw_adjustment" else 45.0
                    target_pos = obj_pos.copy()
                    recovery_log["policy"] = "yaw_rotation"
                    recovery_log["parameters"] = {"yaw_deg": yaw, "target_pos": target_pos.tolist()}
                    obs, frames, lifted_any, target_ok, wrong_ok = perform_grasp_attempt(
                        obs, target_pos, pre_yaw_deg=yaw
                    )

                elif failure_type == "slip_after_contact":
                    target_pos = obj_pos.copy()
                    gt_slip = get_target_object_ground_truth(env, target_obj_name)
                    slip_params = {
                        "extra_depth_below_nominal_com_grasp_m": SLIP_RECOVERY_Z_OFFSET_M,
                        "settle_steps": SLIP_RECOVERY_SETTLE_STEPS,
                    }
                    if gt_slip is not None:
                        z_com = float(np.asarray(gt_slip["com_xyz"], dtype=float)[2])
                        nominal_z = z_com + GRASP_Z_OFFSET_FROM_COM_M
                        target_pos[2] = nominal_z + SLIP_RECOVERY_Z_OFFSET_M
                        slip_params["z_com_m"] = z_com
                        slip_params["nominal_grasp_z_m"] = nominal_z
                        slip_params["slip_grasp_z_m"] = float(target_pos[2])
                    else:
                        target_pos[2] += SLIP_RECOVERY_Z_OFFSET_M
                        slip_params["note"] = (
                            "no sim GT; applied offset to previous obj_pos z (legacy)"
                        )
                    slip_params["target_pos"] = target_pos.tolist()
                    slip_params["descent_steps"] = SLIP_RECOVERY_DESCENT_STEPS
                    recovery_log["policy"] = "lower_grasp_more_settle"
                    recovery_log["parameters"] = slip_params
                    print(
                        f"  Slip recovery: grasp Z target {target_pos[2]:.4f} m "
                        f"({abs(SLIP_RECOVERY_Z_OFFSET_M)*100:.1f} cm below nominal COM grasp), "
                        f"{SLIP_RECOVERY_DESCENT_STEPS} descent steps."
                    )
                    obs, frames, lifted_any, target_ok, wrong_ok = perform_grasp_attempt(
                        obs,
                        target_pos,
                        settle_steps=SLIP_RECOVERY_SETTLE_STEPS,
                        descent_steps=SLIP_RECOVERY_DESCENT_STEPS,
                    )

                elif failure_type == "depth_plane_bad":
                    target_pos = obj_pos.copy()
                    target_pos[2] -= 0.02
                    recovery_log["policy"] = "adjust_depth"
                    recovery_log["parameters"] = {"z_offset_m": -0.02, "target_pos": target_pos.tolist()}
                    obs, frames, lifted_any, target_ok, wrong_ok = perform_grasp_attempt(obs, target_pos)

                else:  # abort_unrecoverable
                    print("  Recovery: abort_unrecoverable — no retry possible.")
                    recovery_log["policy"] = "abort"
                    lifted_any, target_ok, wrong_ok = False, False, False

                if target_pos is not None:
                    obj_pos = target_pos.copy()

                if target_ok:
                    print(f"\nOutcome: RECOVERY SUCCESS on Attempt {metrics['attempts']}")
                    metrics["recovery_success"] = True
                    metrics["task_success"] = True
                    metrics["grasp_success"] = True
                    recovery_log["outcome"] = "success"
                else:
                    print(f"\nOutcome: RECOVERY FAILURE on Attempt {metrics['attempts']}")
                    recovery_log["outcome"] = "failure"
                    if wrong_ok or lifted_any:
                        metrics["wrong_object"] = True
                        metrics["grasp_success"] = True

                try:
                    with open(os.path.join(attempt_dir, "recovery_action.json"), "w") as f:
                        _json.dump(recovery_log, f, indent=2)
                except Exception:
                    pass

                if target_ok or recovery_log["policy"] == "abort":
                    break
        
    if video_enabled and video_writer:
        video_writer.close()
        print(f"Video saved → {current_vid_path}")

    env.close()
    metrics["latency"] = time.time() - start_time

    # Always write a trial summary so every run has a complete record
    try:
        import json as _json
        with open(os.path.join(run_dir, "trial_summary.json"), "w") as f:
            _json.dump({
                "condition": condition,
                "instruction": instruction,
                "trial_idx": trial_idx,
                "run_dir": run_dir,
                "cereal_episode_pose_m": {
                    "pos": np.asarray(episode_cereal_pose["pos"], dtype=float).tolist(),
                    "yaw_deg": float(episode_cereal_pose["yaw_deg"]),
                },
                "baseline_randomize_cereal": bool(randomize_cereal),
                **metrics,
            }, f, indent=2)
        print(f"Trial summary → {os.path.join(run_dir, 'trial_summary.json')}")

        tuning_payload = {
            "run_dir": run_dir,
            "target_object": target_obj_name,
            "cereal_episode_pose": {
                "pos": np.asarray(episode_cereal_pose["pos"], dtype=float).tolist(),
                "yaw_deg": float(episode_cereal_pose["yaw_deg"]),
            },
            "baseline_randomize_cereal": bool(randomize_cereal),
            "tunable_constants": snapshot_tunable_constants(),
            "initial_detection_vs_gt": initial_detection_vs_gt,
            "per_grasp_contact_snapshot": tuning_grasp_records,
            "notes": [
                "commanded_target_z uses a hardcoded table plane after OWL-ViT XY; see delta_commanded_minus_gt_com_m for vertical mismatch vs sim COM.",
                "eef_planar_yaw_minus_object_body_* compares world-XY heading of gripper +X to object body axes; a 'good' grasp may differ by ~90° depending on finger approach vs box edge.",
            ],
        }
        with open(os.path.join(run_dir, "tuning_vs_ground_truth.json"), "w") as f:
            _json.dump(tuning_payload, f, indent=2)
        print(f"Tuning vs GT → {os.path.join(run_dir, 'tuning_vs_ground_truth.json')}")
    except Exception:
        pass

    return metrics

if __name__ == "__main__":
    run_baseline("pick the milk")
