import robosuite as suite
import numpy as np
import torch
from transformers import OwlViTForObjectDetection
import os
import matplotlib.pyplot as plt

def run_baseline_tracking():
    print("Running tracking to generate graph...")
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32").to(device)
    
    env = suite.make(
        env_name="PickPlace", 
        robots="Panda",             
        controller_configs=suite.load_composite_controller_config(controller="BASIC"),
        has_renderer=False,
        use_camera_obs=False,
        has_offscreen_renderer=False,
        control_freq=20,
    )
    obs = env.reset()
    target_obj_name = "Milk"
    
    obj = next((o for o in env.objects if target_obj_name.lower() in o.name.lower()), None)
    
    z_history = []
    obj_z_history = []
    
    def step_towards(current_obs, target_xyz, gripper_action, steps=40):
        body_id = env.sim.model.body_name2id(obj.root_body)
        for _ in range(steps):
            current_eef = current_obs['robot0_eef_pos']
            z_history.append(current_eef[2])
            obj_pos = env.sim.data.body_xpos[body_id]
            obj_z_history.append(obj_pos[2].copy())
            
            delta = target_xyz - current_eef
            action = np.zeros(7)
            action[:3] = delta * 5.0 
            action[6] = gripper_action
            current_obs, reward, done, info = env.step(action)
        return current_obs
        
    body_id = env.sim.model.body_name2id(obj.root_body)
    obj_pos = env.sim.data.body_xpos[body_id].copy()

    # Move above object
    hover_pos = obj_pos.copy()
    hover_pos[2] += 0.2
    obs = step_towards(obs, hover_pos, gripper_action=-1) 
    
    # Lower to grasp
    grasp_pos = obj_pos.copy()
    grasp_pos[2] += 0.02 
    obs = step_towards(obs, grasp_pos, gripper_action=-1, steps=30)
    
    # Close gripper
    obs = step_towards(obs, grasp_pos, gripper_action=1, steps=20) 
    
    # Lift object
    lift_pos = grasp_pos.copy()
    lift_pos[2] += 0.3
    obs = step_towards(obs, lift_pos, gripper_action=1, steps=50)
    
    env.close()

    # Plotting
    plt.figure(figsize=(10,6))
    plt.plot(z_history, label='End-Effector Z-Position', linewidth=2, color='blue')
    plt.plot(obj_z_history, label='Milk Z-Position', linewidth=2, color='red', linestyle='--')
    
    plt.title('End-Effector and Object Height during Grasping Task')
    plt.xlabel('Simulation Steps')
    plt.ylabel('Height (Z-axis) [m]')
    
    # Mark phases
    plt.axvline(x=40, color='gray', linestyle=':', label='Lowering starts')
    plt.axvline(x=70, color='orange', linestyle=':', label='Closing gripper starts')
    plt.axvline(x=90, color='green', linestyle=':', label='Lifting starts')

    plt.legend()
    plt.grid(True, linestyle='-', alpha=0.3)
    plt.savefig('trajectory_plot.png', dpi=300, bbox_inches='tight')
    print("Graph saved as trajectory_plot.png")

if __name__ == "__main__":
    run_baseline_tracking()
