import robosuite as suite
import numpy as np
import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection
import time
import os
import platform
import sys
import datetime
from PIL import Image, ImageDraw
from robosuite.utils.camera_utils import get_camera_transform_matrix, get_real_depth_map, transform_from_pixels_to_world, project_points_from_world_to_camera
import robosuite.utils.transform_utils as T

def simplify_environment(env):
    """Reposition objects to create a 1-bin / 1-object simplified world (Cereal only)."""
    print("--- SIMPLIFYING ENVIRONMENT: 1-Bin / 1-Object Mode (Cereal only) ---")
    try:
        # Hide all objects except Cereal
        for obj_name in ["Bread_joint0", "Can_joint0", "Milk_joint0"]:
            j_id = env.sim.model.joint_name2id(obj_name)
            q_idx = env.sim.model.jnt_qposadr[j_id]
            env.sim.data.qpos[q_idx : q_idx + 3] = np.array([5.0, 5.0, -1.0])

        # Place Cereal in the center of Bin 1 with clearance from all walls
        cereal_id = env.sim.model.joint_name2id("Cereal_joint0")
        cereal_idx = env.sim.model.jnt_qposadr[cereal_id]
        env.sim.data.qpos[cereal_idx : cereal_idx + 3] = np.array([0.08, -0.25, 0.9])

        env.sim.forward()  # Force physics update

        # Let physics settle so GT position is stable before any GT read
        for _ in range(50):
            env.sim.step()
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

def create_composite_image(obs, env):
    """Create a side-by-side (Front + Bird) image for the VLM."""
    front = obs["frontview_image"][::-1]
    bird = obs["birdview_image"][::-1]
    
    # Draw grids and gizmos on both
    front_grid = draw_red_grid_on_array(front)
    front_gizmo = draw_gripper_gizmo(front_grid, obs, env, "frontview")
    
    bird_grid = draw_red_grid_on_array(bird)
    bird_gizmo = draw_gripper_gizmo(bird_grid, obs, env, "birdview")
    
    # Horizontal stack
    composite = np.hstack([front_gizmo, bird_gizmo])
    return composite

def create_frontview_image(obs, env):
    """Create the front-view image used for Gemini localization."""
    front = obs["frontview_image"][::-1]
    front_grid = draw_red_grid_on_array(front)
    return draw_gripper_gizmo(front_grid, obs, env, "frontview")

def draw_target_anchor_on_composite(image_array, front_u, front_v):
    """Mark the last commanded front-view target so Gemini has a visible anchor."""
    image = Image.fromarray(image_array.copy())
    draw = ImageDraw.Draw(image)
    h, w = image_array.shape[:2]
    anchor_x = int(np.clip(round(front_u), 0, w - 1))
    anchor_y = int(np.clip(round(front_v), 0, h - 1))
    cross_len = 10
    color = (0, 255, 255)

    draw.line([(anchor_x - cross_len, anchor_y), (anchor_x + cross_len, anchor_y)], fill=color, width=3)
    draw.line([(anchor_x, anchor_y - cross_len), (anchor_x, anchor_y + cross_len)], fill=color, width=3)
    draw.ellipse([(anchor_x - 4, anchor_y - 4), (anchor_x + 4, anchor_y + 4)], outline=color, width=2)
    draw.text((anchor_x + 8, max(anchor_y - 18, 0)), "LAST TARGET", fill=color)
    return np.array(image)

def draw_gemini_prediction_on_composite(image_array, last_u, last_v, predicted_u, predicted_v):
    """Overlay Gemini's corrected front-view contact point on the image."""
    image = Image.fromarray(image_array.copy())
    draw = ImageDraw.Draw(image)
    h, w = image_array.shape[:2]

    start_x = int(np.clip(round(last_u), 0, w - 1))
    start_y = int(np.clip(round(last_v), 0, h - 1))
    pred_x = int(np.clip(round(predicted_u), 0, w - 1))
    pred_y = int(np.clip(round(predicted_v), 0, h - 1))
    color = (255, 64, 0)

    draw.line([(start_x, start_y), (pred_x, pred_y)], fill=color, width=3)
    draw.ellipse([(pred_x - 6, pred_y - 6), (pred_x + 6, pred_y + 6)], outline=color, width=3)
    draw.text((pred_x + 8, max(pred_y - 18, 0)), "GEMINI TARGET", fill=color)
    return np.array(image)

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

def robust_project_front_pixel_to_world(real_depth, cam_mat, img_h, img_w, u_px, v_px, reference_world=None, search_radius=12, stride=2):
    """
    Back-project a front-view pixel to world coordinates using a local pixel neighborhood.
    This avoids large world-space jumps when the exact corrected pixel lands on an edge or background.
    """
    candidates = []
    center_u = float(np.clip(u_px, 0, img_w - 1))
    center_v = float(np.clip(v_px, 0, img_h - 1))

    for dv in range(-search_radius, search_radius + 1, stride):
        for du in range(-search_radius, search_radius + 1, stride):
            sample_u = float(np.clip(center_u + du, 0, img_w - 1))
            sample_v = float(np.clip(center_v + dv, 0, img_h - 1))
            sample_orig_v = (img_h - 1) - sample_v
            sample_pixels = np.array([sample_orig_v, sample_u])

            try:
                world_pt = transform_from_pixels_to_world(sample_pixels, real_depth, cam_mat)
            except Exception:
                continue

            if not np.all(np.isfinite(world_pt)):
                continue

            pixel_dist = np.hypot(du, dv)
            world_xy_penalty = 0.0
            if reference_world is not None:
                world_xy_penalty = np.linalg.norm(world_pt[:2] - reference_world[:2]) * 250.0
            score = pixel_dist + world_xy_penalty
            candidates.append((score, pixel_dist, world_pt, sample_u, sample_v))

    if not candidates:
        sample_orig_v = (img_h - 1) - center_v
        return transform_from_pixels_to_world(np.array([sample_orig_v, center_u]), real_depth, cam_mat), center_u, center_v

    candidates.sort(key=lambda item: item[0])
    top_world = np.array([item[2] for item in candidates[: min(9, len(candidates))]])
    median_world = np.median(top_world, axis=0)
    best = min(candidates, key=lambda item: np.linalg.norm(item[2][:2] - median_world[:2]) + item[1] * 0.1)
    return best[2], best[3], best[4]

def apply_decoupled_pixel_update(real_depth, cam_mat, img_h, img_w, last_u, last_v, new_u, new_v, reference_world):
    """
    Convert Gemini's image-space correction into a world-space target by treating u and v
    as separate updates. Each image axis only updates its dominant planar world axis, while
    the other world dimensions stay anchored to the previous target.
    """
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

    if abs(delta_u_world[0]) >= abs(delta_u_world[1]):
        u_axis = 0
    else:
        u_axis = 1

    # Force v to control the other planar axis so the image axes stay decoupled.
    v_axis = 1 - u_axis

    new_world = reference_world.copy()
    new_world[u_axis] = reference_world[u_axis] + delta_u_world[u_axis]
    new_world[v_axis] = reference_world[v_axis] + delta_v_world[v_axis]

    return new_world, {
        "base_used_u": base_used_u,
        "base_used_v": base_used_v,
        "used_u": used_u,
        "used_v": used_v,
        "u_axis": u_axis,
        "v_axis": v_axis,
        "delta_u_world_x": float(delta_u_world[0]),
        "delta_u_world_y": float(delta_u_world[1]),
        "delta_v_world_x": float(delta_v_world[0]),
        "delta_v_world_y": float(delta_v_world[1]),
    }

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
        "explanation": ""
    }
    
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
    render_enabled = os.environ.get("BASELINE_RENDER", "0") == "1"
    IMG_H, IMG_W = 512, 512
    video_writer = None
    vid_path = None
    current_vid_path = None
    if render_enabled:
        timestamp = datetime.datetime.now().strftime("%m%d_%I_%M_%p").lower()
        run_name = f"run_{condition}_trial_{trial_idx}_{timestamp}"
        run_dir = os.path.join("runs", run_name)
        os.makedirs(run_dir, exist_ok=True)
        vid_path = os.path.join(run_dir, "attempt_1_run.mp4")
        
        print(f"Rendering requested. Generating video '{vid_path}'...")
        import imageio
        video_writer = imageio.get_writer(vid_path, fps=20)

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

    POS_GAIN = 3.4
    MAX_CART_ACTION = 0.45
    POS_TOL = 0.005
    MAX_YAW_ACTION = 0.18
    YAW_TOL = np.deg2rad(1.5)

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
            if render_enabled and video_writer:
                video_writer.append_data(current_obs["frontview_image"][::-1])

        if settle:
            settle_action = np.zeros(7)
            settle_action[6] = gripper_action
            current_obs, reward, done, info = env.step(settle_action)
            if render_enabled and video_writer:
                video_writer.append_data(current_obs["frontview_image"][::-1])
        return current_obs

    def rotate_yaw_in_place(current_obs, yaw_delta_rad, gripper_action, steps=20):
        if abs(yaw_delta_rad) < YAW_TOL:
            return current_obs

        remaining = float(yaw_delta_rad)
        for _ in range(steps):
            if abs(remaining) < YAW_TOL:
                break

            action = np.zeros(7)
            action[5] = np.clip(remaining * 0.35, -MAX_YAW_ACTION, MAX_YAW_ACTION)
            action[6] = gripper_action
            current_obs, _, _, _ = env.step(action)
            remaining -= action[5]
            if render_enabled and video_writer:
                video_writer.append_data(current_obs["frontview_image"][::-1])

        return current_obs

    # 3. Task Execution
    obs = env.reset()
    
    # SIMPLIFY: Re-position objects into 1-bin / 2-object configuration
    simplify_environment(env)
    
    # CLEAR VIEW: retract along the same smoothed motion profile used for grasping.
    print("--- CLEAR VIEW: Retracting arm for initial perception ---")
    retract_pos = np.array([0.4, -0.6, 1.4])
    obs = step_towards(obs, retract_pos, gripper_action=-1, steps=12)
    owlvit_frontview = create_frontview_image(obs, env)
    if render_enabled and run_dir:
        try:
            Image.fromarray(owlvit_frontview).save(os.path.join(run_dir, "owlvit_clear_view.png"))
        except Exception:
            pass
    
    if render_enabled and video_writer:
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
        if render_enabled and video_writer: video_writer.close()
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

    body_map = {"Cereal": "Cereal_main", "Milk": "Milk_main", "Bread": "Bread_main", "Can": "Can_main"}
    sim_body_name = body_map.get(target_obj_name, target_obj_name)
    pt_3d[2] = 0.825
    print("Using OWL-ViT 2D detection + depth projection only for the initial grasp target.")

    # Store for divergence fallback in the retry loop
    initial_owlvit_3d = pt_3d.copy()

    print(f"Targeting '{target_obj_name}' at 3D coordinate {pt_3d}...")

    # VLM Target 3D Position
    obj_pos = pt_3d.copy()

    HOME_POS = np.array([0.4, -0.6, 1.4])

    # Move above object
    print("Action Plan: Hovering above object...")
    hover_pos = obj_pos.copy()
    hover_pos[2] += 0.2
    obs = step_towards(obs, hover_pos, gripper_action=-1, steps=10) # Open gripper (-1)
    
    # Move down to object center (Z from GT so gripper wraps the object, not the floor)
    print("Action Plan: Lowering to grasp...")
    grasp_pos = obj_pos.copy()
    obs = step_towards(obs, grasp_pos, gripper_action=-1, steps=8)
    
    # Close gripper
    print("Action Plan: Closing gripper...")
    obs = step_towards(obs, grasp_pos, gripper_action=1, steps=6, settle=True) # Close gripper (1)
    last_contact_world = obs['robot0_eef_pos'].copy()
    
    # Lift object
    print("Action Plan: Lifting...")
    lift_pos = grasp_pos.copy()
    lift_pos[2] += 0.3
    obs = step_towards(obs, lift_pos, gripper_action=1, steps=10)
    
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

    lifted_any, target_picked, wrong_picked = check_objects_lifted(env, target_obj_name)
    
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
            
        if condition in ["explanation_only", "feedback", "feedback_double", "feedback_6"]:
            max_retries = 1 if condition == "feedback" else (2 if condition == "feedback_double" else (2 if condition == "feedback_6" else 0))
            error_history = []  # Fix 6: track XY error across attempts for divergence detection

            for retry_idx in range(max_retries + 1):
                # Fix 2+3: Full retract FIRST so depth and composite are from the same clean unoccluded frame
                print("--- FULL RETRACT: Arm to start position for clean analysis ---")
                obs = step_towards(obs, HOME_POS, gripper_action=-1, steps=10)

                # Capture clean depth and composite from the same retracted frame
                clean_depth = obs["frontview_depth"].copy()
                clean_frontview = create_frontview_image(obs, env)

                # Use the actual grasp-contact point, not the commanded target, as Gemini's correction anchor.
                cam_transform_fv = get_camera_transform_matrix(env.sim, "frontview", IMG_H, IMG_W)
                anchor_world = last_contact_world if 'last_contact_world' in locals() else obj_pos
                anchor_px = project_points_from_world_to_camera(anchor_world.reshape(1, 3), cam_transform_fv, IMG_H, IMG_W)[0]
                last_grasp_v_px, last_grasp_u_px = float(anchor_px[0]), float(anchor_px[1])
                gemini_composite = draw_target_anchor_on_composite(clean_frontview, last_grasp_u_px, last_grasp_v_px)
                try:
                    gt_pos = env.sim.data.get_body_xpos(sim_body_name)
                    error_history.append(float(np.linalg.norm((obj_pos - gt_pos)[:2])))
                except Exception:
                    pass

                try:
                    from src.explanation_module import analyze_failure, OPENROUTER_MODEL

                    print(
                        f"Querying {OPENROUTER_MODEL} with clean retracted view "
                        f"(last grasp anchor: u={last_grasp_u_px:.0f} v={last_grasp_v_px:.0f})..."
                    )
                    explanation_json = analyze_failure(target_obj_name, gemini_composite, last_grasp_u_px, last_grasp_v_px)

                    if explanation_json:
                        if render_enabled and run_dir:
                            try:
                                import json
                                log_path = os.path.join(run_dir, f"llm_log_failure_{retry_idx+1}.json")
                                with open(log_path, "w") as f:
                                    json.dump(explanation_json, f, indent=2)

                                Image.fromarray(gemini_composite).save(os.path.join(run_dir, f"llm_input_composite_{retry_idx+1}.png"))
                            except Exception: pass

                        metrics["failure_type"] = explanation_json.get("failure_type", "")
                        metrics["explanation"] = explanation_json.get("explanation", "")

                        print(f"\n[GEMINI FEEDBACK JSON]")
                        print(f"Failure Type: {explanation_json.get('failure_type')}")
                        print(f"Explanation: {explanation_json.get('explanation')}")
                        print(f"Action: {explanation_json.get('suggested_action')} | delta_u: {explanation_json.get('delta_u')} delta_v: {explanation_json.get('delta_v')} yaw: {explanation_json.get('suggested_yaw_delta_deg')}deg")

                        if retry_idx < max_retries and str(explanation_json.get("suggested_action", "")).lower().strip() == "retry":
                            metrics["attempts"] += 1
                            print(f"\n--- INITIATING ATTEMPT {metrics['attempts']} ---")

                            if render_enabled and video_writer is not None:
                                try:
                                    video_writer.close()
                                except Exception: pass
                                current_vid_path = os.path.join(run_dir, f"attempt_{metrics['attempts']}_run.mp4")
                                try:
                                    import imageio
                                    video_writer = imageio.get_writer(current_vid_path, fps=20)
                                except Exception: pass

                            # Fix 6: Divergence check — fall back to OWL-ViT if XY error grew over last 2 attempts
                            diverged = len(error_history) >= 3 and error_history[-1] > error_history[-3]
                            if diverged:
                                print("--- DIVERGENCE DETECTED: XY error is growing. Falling back to initial OWL-ViT target ---")
                                pt_3d_new = initial_owlvit_3d.copy()
                                u_new = last_grasp_u_px
                                v_new = last_grasp_v_px
                            else:
                                # Fix 1: Apply signed pixel delta anchored to last commanded grasp pixel
                                object_center_u = explanation_json.get("object_center_u")
                                object_center_v = explanation_json.get("object_center_v")
                                if object_center_u is not None and object_center_v is not None:
                                    u_new = float(object_center_u)
                                    v_new = float(object_center_v)
                                    delta_u = u_new - last_grasp_u_px
                                    delta_v = last_grasp_v_px - v_new
                                else:
                                    delta_u = float(explanation_json.get("delta_u", 0.0))
                                    delta_v = float(explanation_json.get("delta_v", 0.0))
                                    u_new = last_grasp_u_px + delta_u
                                    v_new = last_grasp_v_px - delta_v

                                # Fix 2+3: Back-project using clean depth from the same frame Gemini analyzed
                                real_depth_clean = get_real_depth_map(env.sim, clean_depth)
                                cam_mat_clean = np.linalg.inv(cam_transform_fv)
                                pt_3d_new, projection_info = apply_decoupled_pixel_update(
                                    real_depth_clean,
                                    cam_mat_clean,
                                    IMG_H,
                                    IMG_W,
                                    last_grasp_u_px,
                                    last_grasp_v_px,
                                    u_new,
                                    v_new,
                                    obj_pos,
                                )
                                used_u = projection_info["used_u"]
                                used_v = projection_info["used_v"]

                                # Fix 4: Keep X/Y from projection, force Z to initial grasp contact level
                                pt_3d_new[2] = initial_owlvit_3d[2]
                                xy_shift = np.linalg.norm(pt_3d_new[:2] - obj_pos[:2])
                                if xy_shift > 0.08:
                                    print(f"Projected retry target jumped {xy_shift:.3f}m; clamping to preserve locality.")
                                    direction = pt_3d_new[:2] - obj_pos[:2]
                                    pt_3d_new[:2] = obj_pos[:2] + direction / max(np.linalg.norm(direction), 1e-8) * 0.08

                                print(
                                    f"Gemini requested delta_u={delta_u:.1f}, delta_v={delta_v:.1f}; "
                                    f"using front pixel u={used_u:.1f}, v={used_v:.1f} for world projection."
                                )

                            if render_enabled and run_dir:
                                try:
                                    prediction_overlay = draw_gemini_prediction_on_composite(
                                        gemini_composite,
                                        last_grasp_u_px,
                                        last_grasp_v_px,
                                        u_new,
                                        v_new,
                                    )
                                    Image.fromarray(prediction_overlay).save(
                                        os.path.join(run_dir, f"llm_result_overlay_{retry_idx+1}.png")
                                    )

                                    debug_lines = [
                                        f"last_target_u={last_grasp_u_px:.3f}",
                                        f"last_target_v={last_grasp_v_px:.3f}",
                                        f"gemini_object_center_u={u_new:.3f}",
                                        f"gemini_object_center_v={v_new:.3f}",
                                        f"delta_u={delta_u:.3f}",
                                        f"delta_v={delta_v:.3f}",
                                    ]
                                    if 'used_u' in locals() and 'used_v' in locals():
                                        debug_lines.extend([
                                            f"base_used_projection_u={projection_info['base_used_u']:.3f}",
                                            f"base_used_projection_v={projection_info['base_used_v']:.3f}",
                                            f"used_projection_u={used_u:.3f}",
                                            f"used_projection_v={used_v:.3f}",
                                            f"u_updates_world_axis={projection_info['u_axis']}",
                                            f"v_updates_world_axis={projection_info['v_axis']}",
                                            f"delta_u_world_x={projection_info['delta_u_world_x']:.6f}",
                                            f"delta_u_world_y={projection_info['delta_u_world_y']:.6f}",
                                            f"delta_v_world_x={projection_info['delta_v_world_x']:.6f}",
                                            f"delta_v_world_y={projection_info['delta_v_world_y']:.6f}",
                                        ])
                                    debug_lines.extend([
                                        f"projected_world_x={pt_3d_new[0]:.6f}",
                                        f"projected_world_y={pt_3d_new[1]:.6f}",
                                        f"projected_world_z={pt_3d_new[2]:.6f}",
                                    ])
                                    try:
                                        gt_pos = env.sim.data.get_body_xpos(sim_body_name)
                                        debug_lines.extend([
                                            f"ground_truth_world_x={gt_pos[0]:.6f}",
                                            f"ground_truth_world_y={gt_pos[1]:.6f}",
                                            f"ground_truth_world_z={gt_pos[2]:.6f}",
                                        ])
                                    except Exception:
                                        pass
                                    with open(os.path.join(run_dir, f"projection_debug_{retry_idx+1}.txt"), "w") as f:
                                        f.write("\n".join(debug_lines) + "\n")
                                except Exception:
                                    pass

                            print(f"Targeting '{target_obj_name}' at NEW 3D coordinate {pt_3d_new}...")
                            obj_pos = pt_3d_new.copy()

                            # Fix 5: Extract yaw correction from Gemini
                            yaw_delta_rad = np.deg2rad(float(explanation_json.get("suggested_yaw_delta_deg", 0.0)))

                            # Reset to home (left corner) so every retry starts from the same position
                            print("Retry Action Plan: Moving directly to hover...")

                            print("Retry Action Plan: Hovering...")
                            hover_pos = obj_pos.copy()
                            hover_pos[2] += 0.2
                            obs = step_towards(obs, hover_pos, gripper_action=-1, steps=10)

                            # Fix 5: Rotate gripper in place at hover height before descending
                            if abs(yaw_delta_rad) > 0.01:
                                print(f"Retry Action Plan: Rotating gripper {np.rad2deg(yaw_delta_rad):.1f} deg...")
                                obs = rotate_yaw_in_place(obs, yaw_delta_rad, gripper_action=-1, steps=20)

                            print("Retry Action Plan: Lowering...")
                            grasp_pos = obj_pos.copy()
                            obs = step_towards(obs, grasp_pos, gripper_action=-1, steps=8)

                            print("Retry Action Plan: Grasping...")
                            obs = step_towards(obs, grasp_pos, gripper_action=1, steps=6, settle=True)
                            last_contact_world = obs['robot0_eef_pos'].copy()

                            print("Retry Action Plan: Lifting...")
                            lift_pos = grasp_pos.copy()
                            lift_pos[2] += 0.3
                            obs = step_towards(obs, lift_pos, gripper_action=1, steps=10)

                            lifted_any_chk, target_picked_chk, wrong_picked_chk = check_objects_lifted(env, target_obj_name)

                            if target_picked_chk:
                                print(f"\nOutcome: RECOVERY SUCCESS on Attempt {metrics['attempts']}")
                                metrics["recovery_success"] = True
                                metrics["task_success"] = True
                                metrics["grasp_success"] = True
                                break
                            else:
                                print(f"\nOutcome: RECOVERY FAILURE on Attempt {metrics['attempts']}")
                                if lifted_any_chk:
                                    metrics["wrong_object"] = True
                                    metrics["grasp_success"] = True
                        else:
                            print(f"\nClosing feedback loop: Gemini did not suggest 'retry'.")
                            break
                    else:
                        print("Gemini returned no JSON. Ending trial.")
                        break
                except Exception as e:
                    print(f"Error in feedback loop: {e}")
                    break
        
    if render_enabled and video_writer:
        video_writer.close()
        print(f"Final video saved to '{current_vid_path if current_vid_path else vid_path}'.")
        
    env.close()
    metrics["latency"] = time.time() - start_time
    
    return metrics

if __name__ == "__main__":
    run_baseline("pick the milk")
