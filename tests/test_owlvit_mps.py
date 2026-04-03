import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection
import numpy as np

def test_owlvit():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("Loading OWL-ViT model (small)...")
    processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
    model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32").to(device)
    
    # Create a dummy image mimicking robosuite output
    image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    texts = [["a photo of a red mug", "a photo of a blue block"]]
    
    print("Processing inputs...")
    inputs = processor(text=texts, images=image, return_tensors="pt").to(device)
    
    print("Running inference...")
    with torch.no_grad():
        outputs = model(**inputs)
    
    target_sizes = torch.tensor([image.shape[:2]]).to(device)
    results = processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0.1)
    
    i = 0  # Batch index
    text = texts[i]
    boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]
    
    print(f"Detected {len(boxes)} boxes.")
    print("OWL-ViT inference test with MPS completed successfully.")

if __name__ == "__main__":
    test_owlvit()
