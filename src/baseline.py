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
from PIL import Image, ImageDraw
from robosuite.utils.camera_utils import get_camera_transform_matrix, get_real_depth_map, transform_from_pixels_to_world, project_points_from_world_to_camera
import robosuite.utils.transform_utils as T

CEREAL_BOX_YAW_DEG = -45.0
POS_GAIN = 3.4
MAX_CART_ACTION = 0.45
POS_TOL = 0.005
MAX_YAW_ACTION = 0.18
YAW_TOL = np.deg2rad(1.5)
DEFAULT_GRIPPER_YAW_DEG = -45.0
OCCLUSION_OBS_POS = np.array([0.22, -0.35, 1.28])

def simplify_environment(env):
    """Reposition objects to create a 1-bin / 1-object simplified world (Cereal only)."""
    print("--- SIMPLIFYING ENVIRONMENT: 1-Bin / 1-Object Mode (Cereal only) ---")
    try:
        cereal_pos = np.array([0.08, -0.25, 0.9])
        half_yaw_rad = np.deg2rad(CEREAL_BOX_YAW_DEG) / 2.0
        cereal_quat = np.array([np.cos(half_yaw_rad), 0.0, 0.0, np.sin(half_yaw_rad)])

        # Hide all objects except Cereal
        for obj_name in ["Bread_joint0", "Can_joint0", "Milk_joint0"]:
            j_id = env.sim.model.joint_name2id(obj_name)
            q_idx = env.sim.model.jnt_qposadr[j_id]
            env.sim.data.qpos[q_idx : q_idx + 3] = np.array([5.0, 5.0, -1.0])

        # Place Cereal in the center of Bin 1 with clearance from all walls
        cereal_id = env.sim.model.joint_name2id("Cereal_joint0")
        cereal_idx = env.sim.model.jnt_qposadr[cereal_id]
        env.sim.data.qpos[cereal_idx : cereal_idx + 3] = cereal_pos
        env.sim.data.qpos[cereal_idx + 3 : cereal_idx + 7] = cereal_quat

        env.sim.forward()  # Force physics update

        # Let physics settle so GT position is stable before any GT read
        for _ in range(50):
            env.sim.step()

        env.sim.data.qpos[cereal_idx : cereal_idx + 3] = cereal_pos
        env.sim.data.qpos[cereal_idx + 3 : cereal_idx + 7] = cereal_quat
        qvel_idx = env.sim.model.jnt_dofadr[cereal_id]
        env.sim.data.qvel[qvel_idx : qvel_idx + 6] = 0.0
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

def get_max_attempts_for_condition(condition):
    if condition == "feedback":
        return 2
    if condition == "feedback_double":
        return 3
    match = re.fullmatch(r"feedback_(\d+)", condition or "")
    if match:
        return max(1, int(match.group(1)))
    return 1

def run_baseline(instruction="pick the milk", condition="feedback", trial_idx=0, seed=None, processor=None, model=None, device=None):
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
    
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    # 1. Perception Step (OWL-ViT initialization)
    if device is None:
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    if processor is None or model is None:
        print("Loading OWL-ViT model onto device...")
        processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
        model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32").to(device)

    # 2. Environment Setup
    print("Initializing Robosuite 'PickPlace' environment with Panda arm...")
    IMG_H, IMG_W = 512, 512

    # Always create a timestamped run directory for artifacts and video.
    # Set BASELINE_RENDER=0 to disable video recording (frames + logs still saved).
    import imageio
    timestamp = datetime.datetime.now().strftime("%m%d_%I_%M_%p").lower()
    run_name = f"run_{condition}_trial_{trial_idx}_{timestamp}"
    run_dir = os.path.join("runs", run_name)
    os.makedirs(run_dir, exist_ok=True)
    print(f"Run artifacts → {run_dir}")

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
        target_xyz = np.array(target_xyz, dtype=float)

        for _ in range(steps):
            current_eef = current_obs['robot0_eef_pos']
            delta = target_xyz - current_eef
            if np.linalg.norm(target_xyz - current_eef) < POS_TOL:
                break

            action = np.zeros(7)
            distance = np.linalg.norm(delta)
            speed_scale = 1.0 if distance > 0.10 else (0.7 if distance > 0.04 else 0.45)
            action[:3] = np.clip(delta * POS_GAIN, -MAX_CART_ACTION * speed_scale, MAX_CART_ACTION * speed_scale)
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
    
    # SIMPLIFY: Re-position objects into 1-bin / 2-object configuration
    simplify_environment(env)
    
    # CLEAR VIEW: retract along the same smoothed motion profile used for grasping.
    print("--- CLEAR VIEW: Retracting arm for initial perception ---")
    retract_pos = np.array([0.4, -0.6, 1.4])
    obs = step_towards(obs, retract_pos, gripper_action=-1, steps=12)
    print(f"--- DEFAULT YAW: Rotating gripper to {DEFAULT_GRIPPER_YAW_DEG:.0f}° before detection/grasp ---")
    obs = rotate_yaw_in_place(obs, np.deg2rad(DEFAULT_GRIPPER_YAW_DEG), gripper_action=-1, steps=40)
    owlvit_frontview = create_frontview_image(obs, env)
    try:
        Image.fromarray(owlvit_frontview).save(os.path.join(run_dir, "owlvit_clear_view.png"))
    except Exception:
        pass
    
    if video_enabled and video_writer:
        video_writer.append_data(obs["frontview_image"][::-1])
    
    # Simulate VLM picking out the target from the language instruction
    target_obj_name = None
    if "milk" in instruction.lower(): target_obj_name = "Milk"
    elif "bread" in instruction.lower(): target_obj_name = "Bread"
    elif "cereal" in instruction.lower(): target_obj_name = "Cereal"
    elif "can" in instruction.lower(): target_obj_name = "Can"
    
    if not target_obj_name:
        print("Failure Reasoning: 'wrong-object selection'")
        print("Explanation: The perception model could not map the instruction to an object.")
        return False
        
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

    pt_3d[2] = 0.825
    print("Using OWL-ViT 2D detection + depth projection only for the initial grasp target.")

    initial_owlvit_3d = pt_3d.copy()

    print(f"Targeting '{target_obj_name}' at 3D coordinate {pt_3d}...")

    # VLM Target 3D Position
    obj_pos = pt_3d.copy()

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
            if z_pos > 0.95:
                lifted_any = True
                if target_name.lower() in o.name.lower():
                    target_picked = True
                else:
                    wrong_picked = True
        return lifted_any, target_picked, wrong_picked

    # Temporal evidence frames captured at each grasp checkpoint for Stage 1 classification
    frames = {}

    def perform_grasp_attempt(current_obs, target_pos, settle_steps=6, pre_yaw_deg=0.0):
        attempt_frames = {}

        print("Action Plan: Hovering above object...")
        hover_pos = target_pos.copy()
        hover_pos[2] += 0.2
        current_obs = step_towards(current_obs, hover_pos, gripper_action=-1, steps=10)
        attempt_frames["pre_hover"] = create_frontview_image(current_obs, env)

        if abs(pre_yaw_deg) > 0.5:
            print(f"Action Plan: Rotating gripper {pre_yaw_deg:.0f}° before descent...")
            current_obs = rotate_yaw_in_place(
                current_obs, np.deg2rad(pre_yaw_deg), gripper_action=-1, steps=20
            )

        print("Action Plan: Lowering to grasp...")
        grasp_pos = target_pos.copy()
        current_obs = step_towards(current_obs, grasp_pos, gripper_action=-1, steps=8)
        attempt_frames["contact"] = create_frontview_image(current_obs, env)

        print("Action Plan: Closing gripper...")
        current_obs = step_towards(current_obs, grasp_pos, gripper_action=1, steps=settle_steps, settle=True)
        attempt_frames["post_close"] = create_frontview_image(current_obs, env)

        print("Action Plan: Lifting...")
        lift_pos = grasp_pos.copy()
        lift_pos[2] += 0.3
        current_obs = step_towards(current_obs, lift_pos, gripper_action=1, steps=10)
        attempt_frames["post_lift"] = create_frontview_image(current_obs, env)

        lifted_any, target_ok, wrong_ok = check_objects_lifted(env, target_obj_name)
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

        feedback_enabled = condition == "explanation_only" or condition.startswith("feedback")
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

                try:
                    with open(os.path.join(attempt_dir, "gemini_prompt.txt"), "w") as f:
                        f.write(failure_result.get("prompt_text", ""))
                    log = {k: v for k, v in failure_result.items() if k != "prompt_text"}
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
                    target_pos[2] -= 0.01
                    recovery_log["policy"] = "lower_grasp_more_settle"
                    recovery_log["parameters"] = {
                        "z_offset_m": -0.01,
                        "settle_steps": 12,
                        "target_pos": target_pos.tolist(),
                    }
                    obs, frames, lifted_any, target_ok, wrong_ok = perform_grasp_attempt(
                        obs, target_pos, settle_steps=12
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
                **metrics,
            }, f, indent=2)
        print(f"Trial summary → {os.path.join(run_dir, 'trial_summary.json')}")
    except Exception:
        pass

    return metrics

if __name__ == "__main__":
    run_baseline("pick the milk")
