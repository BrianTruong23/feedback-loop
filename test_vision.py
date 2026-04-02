import robosuite as suite
import numpy as np
import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from robosuite.utils.camera_utils import get_camera_transform_matrix, get_real_depth_map, transform_from_pixels_to_world
from PIL import Image

def test():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print("Loading OWL-ViT model onto device...")
    processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
    model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32").to(device)

    env = suite.make(
        env_name="PickPlace", 
        robots="Panda",             
        controller_configs=suite.load_composite_controller_config(controller="BASIC"),
        has_renderer=False,
        use_camera_obs=True,      
        has_offscreen_renderer=True,
        camera_names="frontview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=True,
        control_freq=20,
    )
    obs = env.reset()
    
    img = obs["frontview_image"] # Note: robosuite images are upside down! img[::-1]
    depth = obs["frontview_depth"]
    
    # Save the raw image
    raw_pil = Image.fromarray((img[::-1]).astype(np.uint8))
    raw_pil.save("test_frontview.png")
    
    # Run OWL-ViT
    texts = [["a photo of a milk carton", "a photo of a bread", "a photo of a cereal box", "a photo of a can"]]
    # We should run it on the correctly oriented image (img[::-1])
    oriented_img = img[::-1]
    inputs = processor(text=texts, images=oriented_img, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    
    target_sizes = torch.tensor([oriented_img.shape[:2]]).to(device)
    results = processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0.1)
    
    print("\nDetection Results:")
    scores = results[0]["scores"].cpu().numpy()
    labels = results[0]["labels"].cpu().numpy()
    boxes = results[0]["boxes"].cpu().numpy()
    
    for score, label, box in zip(scores, labels, boxes):
        print(f"Detected {texts[0][label]} with score {score:.3f} at {box}")
        if label == 0: # milk
            # bbox format is [xmin, ymin, xmax, ymax]
            u = (box[0] + box[2]) / 2.0
            v = (box[1] + box[3]) / 2.0
            print(f"Center pixel: u (x)={u:.2f}, v (y)={v:.2f}")
            
            # Since depth map needs to map to original img map
            # original img has y inverted. So original_v = 256 - v
            orig_v = 256 - v
            
            # Let's project to 3d
            real_depth = get_real_depth_map(env.sim, depth)
            cam_mat = get_camera_transform_matrix(env.sim, "frontview", 256, 256)
            pixels = np.array([[u, orig_v]])
            real_depth_img = np.expand_dims(real_depth, axis=-1) # shape [1, 256, 256, 1] for the batch
            # wait transform_from_pixels_to_world expects pixels [batch, 2] wait, the internal depth is passed as [H, W, 1]?
            
            pt_3d = transform_from_pixels_to_world(pixels, [real_depth_img], cam_mat)
            print("Projected 3D point:", pt_3d)

            # Let's compare with actual god-mode pos
            target_obj_name = "Milk"
            obj = next((o for o in env.objects if target_obj_name.lower() in o.name.lower()), None)
            body_id = env.sim.model.body_name2id(obj.root_body)
            god_pos = env.sim.data.body_xpos[body_id].copy()
            print("Real 3D God Pos:", god_pos)

if __name__ == "__main__":
    test()
