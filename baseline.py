import robosuite as suite
import numpy as np
import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection
import time

def run_baseline(instruction="pick the milk"):
    print(f"--- Starting Baseline Task ---")
    print(f"Language Instruction: '{instruction}'")
    
    # 1. Perception Step (OWL-ViT initialization)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print("Loading OWL-ViT model onto device...")
    processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
    model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32").to(device)

    # 2. Environment Setup
    print("Initializing Robosuite 'PickPlace' environment with Panda arm...")
    # The default PickPlaceSingle loads a Single Object, PickPlace loads 4 (Milk, Bread, Cereal, Can)
    # We will use PickPlace to have clutter.
    env = suite.make(
        env_name="PickPlace", 
        robots="Panda",             
        controller_configs=suite.load_controller_config(default_controller="OSC_POSE"), 
        has_renderer=True,         # Show the window!
        use_camera_obs=False,      # Set to False to speed up simulation loops if not reading camera directly
        has_offscreen_renderer=False,
        control_freq=20,
        render_camera="frontview",
    )

    # 3. Task Execution
    obs = env.reset()
    env.render()
    
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

    # Fetch ground-truth pos (mimicking a PERFECT 2D->3D mapping from the VLM)
    try:
        # Robosuite adds suffix _body to object names in the physics tree sometimes
        # Specifically for PickPlace, the bodies are named "Milk_body", etc.
        body_name = target_obj_name + "_visual" # just checking existence
    except:
        pass

    # Actually in Robosuite, we can loop through the objects in the environment
    # env.objects is a list of object classes.
    obj = next((o for o in env.objects if target_obj_name.lower() in o.name.lower()), None)
    if not obj:
        print("Failure Reasoning: 'target occluded' / not found")
        return False
        
    print(f"Targeting '{obj.name}'...")

    # Grasp Heuristic
    # Move above -> Move down -> Close Gripper -> Move up
    def step_towards(target_xyz, gripper_action, steps=40):
        for _ in range(steps):
            current_eef = obs['robot0_eef_pos']
            delta = target_xyz - current_eef
            # Action space for OSC_POSE is [dx, dy, dz, dax, day, daz, gripper]
            # scaled to max velocity
            action = np.zeros(7)
            action[:3] = delta * 5.0 
            action[6] = gripper_action
            obs, reward, done, info = env.step(action)
            env.render()
            time.sleep(0.01) # slow down so user can see it
        return obs
        
    # Get initial object position. 
    # The sim ID is usually obj.name + "_body" or we can get it from sim data.
    body_id = env.sim.model.body_name2id(obj.root_body)
    obj_pos = env.sim.data.body_xpos[body_id].copy()

    # Move above object
    print("Action Plan: Hovering above object...")
    hover_pos = obj_pos.copy()
    hover_pos[2] += 0.2
    obs = step_towards(hover_pos, gripper_action=-1) # Open gripper (-1)
    
    # Move down to object
    print("Action Plan: Lowering to grasp...")
    grasp_pos = obj_pos.copy()
    grasp_pos[2] += 0.02 # Slightly above center 
    obs = step_towards(grasp_pos, gripper_action=-1, steps=30)
    
    # Close gripper
    print("Action Plan: Closing gripper...")
    obs = step_towards(grasp_pos, gripper_action=1, steps=20) # Close gripper (1)
    
    # Lift object
    print("Action Plan: Lifting...")
    lift_pos = grasp_pos.copy()
    lift_pos[2] += 0.3
    obs = step_towards(lift_pos, gripper_action=1, steps=50)
    
    # Check outcome
    final_pos = env.sim.data.body_xpos[body_id]
    
    if final_pos[2] > 0.85: # roughly table height
        print("\nOutcome: SUCCESS")
        print("The baseline policy successfully completed the language instruction.")
    else:
        print("\nOutcome: FAILURE")
        print("Failure Reasoning: 'grasp instability'")
        print("Explanation: Evaluator detected that the object slipped or was not successfully lifted above the table height.")
        # This is where the feedback loop in week 2 will kick in!
        
    env.close()

if __name__ == "__main__":
    run_baseline("pick the milk")
