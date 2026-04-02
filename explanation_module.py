import os
import json
import base64
import requests
import io
from PIL import Image
from dotenv import load_dotenv

load_dotenv()
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

def encode_image_array_to_base64(img_array):
    """Convert numpy array (RGB) to base64 jpeg."""
    # Convert numpy array to PIL Image
    image = Image.fromarray(img_array)
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{img_str}"

def analyze_failure(target_object, before_img_array, after_img_array):
    """
    Sends before and after simulation images to Gemini 2.5 Flash.
    Forces the response to follow a strict JSON schema.
    """
    if not OPENROUTER_API_KEY:
        print("Warning: OPENROUTER_API_KEY not found in environment variables.")
        return None

    # Encode images
    before_b64 = encode_image_array_to_base64(before_img_array)
    after_b64 = encode_image_array_to_base64(after_img_array)

    prompt = f"""You are a debugging assistant for a language-guided robotics system. 
The system was instructed to grab the: '{target_object}'. 
I am providing you two images:
1. BEFORE image: The initial state of the table.
2. AFTER image: The final state of the table after the robot attempted the grab and returned to its raised position.

Unfortunately, the robotic evaluator reported that the grasp FAILED (the object was not securely lifted).

Analyze the before and after images.
- Did the robot miss the target entirely?
- Did it choose the wrong object?
- Did it bump the object or drop it?

Provide a JSON response using EXACTLY this schema:
{{
  "failure_type": "target occluded" | "grasp instability" | "wrong-object selection" | "no object reached",
  "explanation": "<short natural language explanation of the physical failure>",
  "suggested_action": "retry" | "abort",
  "updated_u": <float>,
  "updated_v": <float>
}}

Important Instructions for updated_u and updated_v:
- The image size is 256x256. 
- (0, 0) is the top-left corner. u is the X coordinate (width), v is the Y coordinate (height).
- IMPORTANT: I have drawn a red structural grid over the images with lines occurring exactly every 32 pixels. The specific coordinate numbers (0, 32, 64...) are stamped in white at the edges.
- Read the grid lines extremely carefully to estimate the center of the {target_object}.
- Look at the AFTER image. Locate the {target_object}. Estimate its exact center pixel coordinate (u, v) using the overlaid grid.
- Return these coordinates as floats.
- Return these coordinates as floats.
"""

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8000",
        "X-Title": "SimRobotics",
    }

    payload = {
        "model": "google/gemini-2.5-flash",
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": before_b64
                        }
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": after_b64
                        }
                    }
                ]
            }
        ]
    }

    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        result = response.json()
        content = result['choices'][0]['message']['content']
        
        # Parse JSON
        data = json.loads(content)
        return data

    except Exception as e:
        print(f"Error querying OpenRouter API: {e}")
        try:
            print(response.text)
        except:
            pass
        return None

if __name__ == "__main__":
    import sys
    print("Testing Explanation Module Standalone...")
    import numpy as np
    
    # Check if a test image exists
    test_image_path = "test_frontview.png"
    if os.path.exists(test_image_path):
        print(f"Found {test_image_path}. Using it as both before and after images for testing.")
        test_img = np.array(Image.open(test_image_path))
        
        print("Sending request to Gemini...")
        result = analyze_failure("Milk", test_img, test_img)
        print(json.dumps(result, indent=2))
    else:
        print("No test_frontview.png found. Run baseline.py to generate simulation files.")
