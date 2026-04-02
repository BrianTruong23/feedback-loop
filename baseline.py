import robosuite as suite
import numpy as np
import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection
import time
import os
import platform
import sys
import datetime
from PIL import Image
from robosuite.utils.camera_utils import get_camera_transform_matrix, get_real_depth_map, transform_from_pixels_to_world

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
    video_writer = None
    vid_path = None
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
        camera_heights=256,
        camera_widths=256,
        camera_depths=True,
        has_offscreen_renderer=True, # Need offscreen to render camera obs
        control_freq=20,
        render_camera="frontview",
    )

    # 3. Task Execution
    obs = env.reset()
    
    # Store the grid version mathematically for the LLM
    before_img = draw_red_grid_on_array(obs["frontview_image"][::-1])
    
    if render_enabled and run_dir:
        try:
            import PIL.Image as Image
            Image.fromarray(before_img).save(os.path.join(run_dir, "initial_grid_view.png"))
        except Exception: pass
        
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
    orig_v = 256 - v
    
    # Project 2D to 3D using real depth map
    real_depth = get_real_depth_map(env.sim, depth)
    cam_mat = np.linalg.inv(get_camera_transform_matrix(env.sim, "frontview", 256, 256))
    
    # transform_from_pixels_to_world inherently expects [row, col] format (i.e. [v, u])
    # so that bilinear interpolation correctly maps im[v, u] and cam_pts traces u*z, v*z.
    pixels = np.array([orig_v, u])
    
    pt_3d = transform_from_pixels_to_world(pixels, real_depth, cam_mat)
    
    print(f"Targeting '{target_obj_name}' at 3D coordinate {pt_3d}...")

    # Grasp Heuristic
    # Move above -> Move down -> Close Gripper -> Move up
    def step_towards(current_obs, target_xyz, gripper_action, steps=40):
        for _ in range(steps):
            current_eef = current_obs['robot0_eef_pos']
            delta = target_xyz - current_eef
            # Action space for OSC_POSE is [dx, dy, dz, dax, day, daz, gripper]
            # scaled to max velocity
            action = np.zeros(7)
            action[:3] = np.clip(delta * 5.0, -1.0, 1.0)
            action[6] = gripper_action
            current_obs, reward, done, info = env.step(action)
            if render_enabled and video_writer:
                video_writer.append_data(current_obs["frontview_image"][::-1])
        return current_obs
        
    # VLM Target 3D Position
    obj_pos = pt_3d.copy()

    # Move above object
    print("Action Plan: Hovering above object...")
    hover_pos = obj_pos.copy()
    hover_pos[2] += 0.2
    obs = step_towards(obs, hover_pos, gripper_action=-1) # Open gripper (-1)
    
    # Move down to object
    print("Action Plan: Lowering to grasp...")
    grasp_pos = obj_pos.copy()
    grasp_pos[2] += 0.02 # Slightly above center 
    obs = step_towards(obs, grasp_pos, gripper_action=-1, steps=30)
    
    # Close gripper
    print("Action Plan: Closing gripper...")
    obs = step_towards(obs, grasp_pos, gripper_action=1, steps=20) # Close gripper (1)
    
    # Lift object
    print("Action Plan: Lifting...")
    lift_pos = grasp_pos.copy()
    lift_pos[2] += 0.3
    obs = step_towards(obs, lift_pos, gripper_action=1, steps=50)
    
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
            
        if condition in ["explanation_only", "feedback", "feedback_double"]:
            max_retries = 1 if condition == "feedback" else (2 if condition == "feedback_double" else 0)
            
            for retry_idx in range(max_retries + 1):
                # Only explanation_only runs once and breaks instantly; feedback runs up to 2, feedback_double up to 3 queries.
                print(f"\n--- TRIGGERING EXPLANATION MODULE (Failure #{retry_idx + 1}) ---")
                after_img = draw_red_grid_on_array(obs["frontview_image"][::-1])
                try:
                    from explanation_module import analyze_failure
                    print(f"Querying Gemini 2.5 Flash to analyze the failure...")
                    explanation_json = analyze_failure(target_obj_name, before_img, after_img)
                    
                    if explanation_json:
                        # Log the LLM output directly into the same folder as the video!
                        if render_enabled and run_dir:
                            try:
                                import json
                                log_path = os.path.join(run_dir, f"llm_log_failure_{retry_idx+1}.json")
                                with open(log_path, "w") as f:
                                    json.dump(explanation_json, f, indent=2)
                                
                                # Export exactly the images analyzed
                                Image.fromarray(before_img).save(os.path.join(run_dir, f"before_image_failure_{retry_idx+1}.png"))
                                Image.fromarray(after_img).save(os.path.join(run_dir, f"after_image_failure_{retry_idx+1}.png"))
                                
                                # Export robot physical XYZ position
                                eef_pos = obs['robot0_eef_pos'].tolist()
                                pos_info = {"robot_x": eef_pos[0], "robot_y": eef_pos[1], "robot_z": eef_pos[2]}
                                with open(os.path.join(run_dir, f"robot_position_failure_{retry_idx+1}.json"), "w") as f:
                                    json.dump(pos_info, f, indent=2)
                                    
                                # Export the target object explicitly
                                with open(os.path.join(run_dir, "target_object.txt"), "w") as f:
                                    f.write(f"Target Object: {target_obj_name}\n")
                            except Exception: pass
                            
                        metrics["failure_type"] = explanation_json.get("failure_type", "")
                        metrics["explanation"] = explanation_json.get("explanation", "")
                        
                        print(f"\n[GEMINI FEEDBACK JSON]")
                        print(f"Failure Type: {explanation_json.get('failure_type')}")
                        print(f"Explanation: {explanation_json.get('explanation')}")
                        print(f"Action: {explanation_json.get('suggested_action')} | Target U: {explanation_json.get('updated_u')} V: {explanation_json.get('updated_v')}")
                        
                        if retry_idx < max_retries and str(explanation_json.get("suggested_action", "")).lower().strip() == "retry":
                            metrics["attempts"] += 1
                            print(f"\n--- INITIATING ATTEMPT {metrics['attempts']} ---")
                            
                            # Cut the video and start a new one for attempt 2 if rendering
                            if render_enabled and video_writer is not None:
                                try:
                                    video_writer.close()
                                except Exception: pass
                                vid_path_n = os.path.join(run_dir, f"attempt_{metrics['attempts']}_run.mp4")
                                try:
                                    import imageio
                                    video_writer = imageio.get_writer(vid_path_n, fps=20)
                                except Exception: pass
                                
                            # Capture strictly synced depth map BEFORE the arm executes the reset sequence
                            # to ensure the depth map perfectly matches the Gemini after_img
                            after_depth = obs["frontview_depth"].copy()
                            
                            # reset arm visually slightly
                            obs = step_towards(obs, hover_pos, gripper_action=-1, steps=20)
                            
                            u_new = float(explanation_json.get("updated_u", 128.0))
                            v_new = float(explanation_json.get("updated_v", 128.0))
                            
                            # Account for numpy array row indexing appropriately
                            orig_v_new = 255 - v_new
                            
                            real_depth_after = get_real_depth_map(env.sim, after_depth)
                            cam_mat_after = np.linalg.inv(get_camera_transform_matrix(env.sim, "frontview", 256, 256))
                            
                            # Inherently transposed to [row, col] format
                            pixels_new = np.array([orig_v_new, u_new])
                            
                            pt_3d_new = transform_from_pixels_to_world(pixels_new, real_depth_after, cam_mat_after)
                            print(f"Targeting '{target_obj_name}' at NEW 3D coordinate {pt_3d_new}...")
                            
                            obj_pos = pt_3d_new.copy()
                            
                            print("Retry Action Plan: Hovering...")
                            hover_pos = obj_pos.copy()
                            hover_pos[2] += 0.2
                            obs = step_towards(obs, hover_pos, gripper_action=-1)
                            
                            print("Retry Action Plan: Lowering...")
                            grasp_pos = obj_pos.copy()
                            grasp_pos[2] += 0.02
                            obs = step_towards(obs, grasp_pos, gripper_action=-1, steps=30)
                            
                            print("Retry Action Plan: Grasping...")
                            obs = step_towards(obs, grasp_pos, gripper_action=1, steps=20)
                            
                            print("Retry Action Plan: Lifting...")
                            lift_pos = grasp_pos.copy()
                            lift_pos[2] += 0.3
                            obs = step_towards(obs, lift_pos, gripper_action=1, steps=50)
                            
                            lifted_any_chk, target_picked_chk, wrong_picked_chk = check_objects_lifted(env, target_obj_name)
                            
                            if target_picked_chk:
                                print(f"\nOutcome: RECOVERY SUCCESS on Attempt {metrics['attempts']}")
                                metrics["recovery_success"] = True
                                metrics["task_success"] = True
                                metrics["grasp_success"] = True
                                break # Success, escape loop
                            else:
                                print(f"\nOutcome: RECOVERY FAILURE on Attempt {metrics['attempts']}")
                                if lifted_any_chk:
                                    metrics["wrong_object"] = True
                                    metrics["grasp_success"] = True
                                # Loops again if retry_idx < max_retries - 1
                        else:
                            break # OpenRouter didn't say retry or max hits reached
                except Exception as e:
                    print(f"Error running Gemini Explanation Module: {e}")
                    break
        
    if render_enabled and video_writer:
        video_writer.close()
        print(f"Video saved to '{vid_path}'.")
        
    env.close()
    metrics["latency"] = time.time() - start_time
    
    return metrics

if __name__ == "__main__":
    run_baseline("pick the milk")
