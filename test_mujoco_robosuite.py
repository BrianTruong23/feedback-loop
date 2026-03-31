import robosuite as suite
import numpy as np

def test_environment():
    print("Initializing Robosuite environment...")
    # Create an environment to test
    env = suite.make(
        env_name="PickPlace",
        robots="Panda",
        has_renderer=False,       # headless mode just to check physics
        has_offscreen_renderer=True, # enable camera rendering
        use_camera_obs=True,      # get pixel observations
    )

    # Reset the environment
    obs = env.reset()
    print("Environment reset successfully. Testing 10 simulation steps...")

    # Take 10 random actions
    for i in range(10):
        action = np.random.uniform(-1, 1, env.action_dim)
        obs, reward, done, info = env.step(action)
    
    # Check if image data is rendered
    camera_obs = obs.get('agentview_image')
    if camera_obs is not None:
        print(f"Success! Camera rendered an image of shape: {camera_obs.shape}")
    else:
        print("Failed to get agentview_image from observation.")

if __name__ == "__main__":
    test_environment()
