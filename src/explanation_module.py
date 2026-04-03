import os
import json
import base64
import requests
import io
from PIL import Image
from dotenv import load_dotenv

load_dotenv()
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_MODEL = "google/gemini-2.5-flash"

def encode_image_array_to_base64(img_array):
    """Convert numpy array (RGB) to base64 jpeg."""
    # Convert numpy array to PIL Image
    image = Image.fromarray(img_array)
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{img_str}"

def analyze_failure(target_object, composite_img_array, last_grasp_u, last_grasp_v):
    """
    Sends a clean retracted composite Front+Bird view to Gemini 2.5 Flash.
    last_grasp_u, last_grasp_v: pixel coords of the last commanded grasp target in the Front View,
    used as an anchor so Gemini reports a signed delta instead of an absolute guess.
    """
    if not OPENROUTER_API_KEY:
        print("Warning: OPENROUTER_API_KEY not found in environment variables.")
        return None

    # Encode composite image
    composite_b64 = encode_image_array_to_base64(composite_img_array)

    prompt = f"""You are a debugging assistant for a language-guided robotics system.
The system was instructed to grab the: '{target_object}'.

I am providing you with a FRONT VIEW image (512x512) captured after the robot arm has fully retracted out of the way.

The robot's last grasp attempt made contact near pixel (u={last_grasp_u:.0f}, v={last_grasp_v:.0f}) in the Front View.
A CYAN crosshair labeled LAST TARGET is drawn at that exact front-view contact point.
For clarity, the LAST TARGET coordinates are:
- LAST TARGET u = {last_grasp_u:.0f}
- LAST TARGET v = {last_grasp_v:.0f}
The gripper gizmo lines (visible in both views before retraction) were:
- RED line: heading of the fingers.
- GREEN line: the axis along which the gripper closes.

Your task:
1. Locate the {target_object} center in the Front View (u 0-512).
2. Compute delta_u = object_center_u - {last_grasp_u:.0f}  (positive = object is to the RIGHT of the last contact point).
3. Compute delta_v = {last_grasp_v:.0f} - object_center_v  (positive = object is ABOVE the last contact point).
4. If you cannot infer a useful yaw correction from the front view alone, set suggested_yaw_delta_deg to 0.

Provide a JSON response using EXACTLY this schema:
{{
  "failure_type": "target occluded" | "grasp instability" | "wrong-object selection" | "no object reached",
  "explanation": "<short natural language explanation of the failure>",
  "suggested_action": "retry" | "abort",
  "object_center_u": <float, absolute front-view u coordinate of the {target_object} center, 0-512>,
  "object_center_v": <float, absolute front-view v coordinate of the {target_object} center, 0-512>,
  "delta_u": <float, signed pixels to shift RIGHT from the last contact point ({last_grasp_u:.0f}) to object center>,
  "delta_v": <float, signed pixels to shift UP from the last contact point ({last_grasp_v:.0f}) to object center>,
  "suggested_yaw_delta_deg": <float, degrees to rotate gripper: positive=clockwise, negative=counterclockwise, 0 if no rotation needed>,
  "confidence": <float from 0.0 to 1.0 for how certain you are about the target center>
}}

Important Instructions:
- The image is 512x512 and shows only the Front View.
- Use the red grid lines and labels (0, 32, 64, 96... 512) to measure pixel distances carefully.
- The CYAN crosshair labeled LAST TARGET marks the exact previous front-view contact point.
- delta_u and delta_v are RELATIVE offsets from that contact point ({last_grasp_u:.0f}, {last_grasp_v:.0f}), NOT absolute coordinates.
- object_center_u and object_center_v must be the absolute front-view center coordinates of the {target_object}.
- Positive delta_u means move RIGHT in the image. Negative delta_u means move LEFT in the image.
- Example: if LAST TARGET is at u=170 and the object center is at u=210, then delta_u = 210 - 170 = +40, which means 40 pixels RIGHT.
- Example: if LAST TARGET is at u=170 and the object center is at u=150, then delta_u = 150 - 170 = -20, which means 20 pixels LEFT.
- Positive delta_v means move UP in the image. Negative delta_v means move DOWN in the image.
- Example: if LAST TARGET is at v=260 and the object center is at v=230, then delta_v = 260 - 230 = +30, which means 30 pixels UP.
- Example: if LAST TARGET is at v=260 and the object center is at v=290, then delta_v = 260 - 290 = -30, which means 30 pixels DOWN.
- If delta_v is negative, the target is BELOW the cyan LAST TARGET point, so move DOWN.
- Only use the Front View for localization.
- VISUAL CUES for {target_object}:
  * Milk: White/red carton.
  * Cereal: Red box. The red box in the scene is the cereal box.
  * Bread: Tan loaf box.
  * Can: Red cylinder.
"""

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8000",
        "X-Title": "SimRobotics",
    }

    payload = {
        "model": OPENROUTER_MODEL,
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
                            "url": composite_b64
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
    print("Testing Explanation Module Standalone...")
    import numpy as np

    # Check if a test image exists
    test_image_path = "test_frontview.png"
    if os.path.exists(test_image_path):
        print(f"Found {test_image_path}. Using it as test image.")
        test_img = np.array(Image.open(test_image_path))

        print("Sending request to Gemini...")
        result = analyze_failure("Milk", test_img, last_grasp_u=256.0, last_grasp_v=256.0)
        print(json.dumps(result, indent=2))
    else:
        print("No test_frontview.png found. Run baseline.py to generate simulation files.")
