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

FAILURE_TAXONOMY = [
    "wrong_object",
    "no_object_reached",
    "grasp_pose_bad",
    "slip_after_contact",
    "target_occluded",
    "needs_yaw_adjustment",
    "depth_plane_bad",
    "abort_unrecoverable",
]

GRASP_CHECKPOINTS = [
    "target_identified",
    "gripper_aligned_above_target",
    "fingers_enclosing_target",
    "object_lifted_clear_of_bin",
    "correct_object_in_gripper",
]

OBJECT_TASK_DESCRIPTORS = {
    "Milk": {
        "noun": "milk carton",
        "alignment_target": "the milk carton center",
        "straddle_target": "the milk carton side faces",
        "closure_target": "the milk carton side faces",
        "lift_target": "the milk carton",
    },
    "Bread": {
        "noun": "bread loaf box",
        "alignment_target": "the bread loaf box center",
        "straddle_target": "the bread loaf box side faces",
        "closure_target": "the bread loaf box side faces",
        "lift_target": "the bread loaf box",
    },
    "Cereal": {
        "noun": "cereal box",
        "alignment_target": "the cereal box center",
        "straddle_target": "the cereal box side faces",
        "closure_target": "the cereal box side faces",
        "lift_target": "the cereal box",
    },
    "Can": {
        "noun": "soda can",
        "alignment_target": "the soda can center",
        "straddle_target": "the soda can left and right sides",
        "closure_target": "the soda can curved side faces",
        "lift_target": "the soda can",
    },
}


def _encode_image(img_array):
    image = Image.fromarray(img_array)
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


def _build_object_specific_subtasks(target_object):
    spec = OBJECT_TASK_DESCRIPTORS.get(
        target_object,
        {
            "noun": target_object.lower(),
            "alignment_target": f"the {target_object.lower()} center",
            "straddle_target": f"the {target_object.lower()} sides",
            "closure_target": f"the {target_object.lower()} side faces",
            "lift_target": f"the {target_object.lower()}",
        },
    )
    return "\n".join(
        [
            f"  - subtask_1: align gripper above {spec['alignment_target']}",
            f"  - subtask_2: descend so fingers straddle {spec['straddle_target']}",
            f"  - subtask_3: close gripper around {spec['closure_target']}",
            f"  - subtask_4: lift {spec['lift_target']} clear of bin",
        ]
    )


def classify_failure(target_object, frames):
    """
    Stage 1: Classify a grasp failure using 5 sequential front-view frames.

    frames: dict with numpy array values for keys:
        'pre_hover'  - arm hovering above target, gripper open, before descent
        'contact'    - gripper at grasp height, before closing
        'post_close' - gripper closed, attempting to hold object
        'post_lift'  - after lift attempt (failure becomes visible here)
        'retracted'  - clean scene after arm fully retracted to home

    Returns: {
        'failure_type': str (one of FAILURE_TAXONOMY),
        'failed_checkpoint': str (one of GRASP_CHECKPOINTS),
        'explanation': str,
        'confidence': float
    } or None on error.
    """
    if not OPENROUTER_API_KEY:
        print("Warning: OPENROUTER_API_KEY not found.")
        return None

    taxonomy_str = "\n".join(f"  - {t}" for t in FAILURE_TAXONOMY)
    checkpoint_str = "\n".join(f"  - {c}" for c in GRASP_CHECKPOINTS)
    object_specific_subtasks = _build_object_specific_subtasks(target_object)

    prompt = f"""You are a robotics failure analyst. A robot arm attempted to pick up '{target_object}' and FAILED (the object was not successfully lifted).

I am providing 5 consecutive front-view frames (512x512, red pixel-coordinate grid overlay) that capture the full grasp sequence:

  Frame 1 — PRE-HOVER: Arm at hover height above target, gripper open, before descent.
  Frame 2 — CONTACT: Gripper lowered to grasp height, before closing.
  Frame 3 — POST-CLOSE: Gripper has closed (attempting to hold object).
  Frame 4 — POST-LIFT: After the lift attempt — this is where the failure is most visible.
  Frame 5 — RETRACTED: Clean scene after arm fully retracted to home position.

Reason about the attempt through these object-specific sub-tasks:
{object_specific_subtasks}

STEP 1 — Checkpoint analysis.
Examine the frames in sequence and determine the FIRST checkpoint that failed. Use the object-specific sub-tasks above to ground your decision:
{checkpoint_str}

STEP 2 — Failure classification.
Based on the failed checkpoint, classify the failure into EXACTLY ONE of these types:
{taxonomy_str}

Definitions:
  - wrong_object: The robot targeted or grasped a different object than '{target_object}'.
  - no_object_reached: The gripper descended but clearly missed the object entirely (no meaningful contact).
  - grasp_pose_bad: Gripper made contact but was off-center or poorly positioned for a stable grasp.
  - slip_after_contact: Gripper appeared to hold the object (POST-CLOSE shows contact) but it slipped during lift.
  - target_occluded: The target was not clearly visible in the scene, preventing reliable detection.
  - needs_yaw_adjustment: Gripper contacted the object but its rotational orientation (yaw) was wrong for enclosure.
  - depth_plane_bad: Gripper descended to the wrong depth (too shallow — barely touched, or too deep — pushed through).
  - abort_unrecoverable: Object fell out of bin, severe scene disruption, or no single-retry fix exists.

Respond with EXACTLY this JSON (no markdown, no extra keys):
{{
  "failed_checkpoint": "<one of the 5 checkpoints above>",
  "failure_type": "<one of the 8 types above>",
  "explanation": "<1-2 sentences describing what you observe across the frames>",
  "confidence": <float 0.0 to 1.0>
}}"""

    frame_order = ["pre_hover", "contact", "post_close", "post_lift", "retracted"]
    frame_labels = ["PRE-HOVER", "CONTACT", "POST-CLOSE", "POST-LIFT", "RETRACTED"]
    extra_keys = [key for key in frames.keys() if key not in frame_order]
    ordered_keys = frame_order + sorted(extra_keys)

    content = [{"type": "text", "text": prompt}]
    for i, key in enumerate(ordered_keys):
        if key in frames and frames[key] is not None:
            if key in frame_order:
                label = frame_labels[frame_order.index(key)]
            else:
                label = key.replace("_", " ").upper()
            content.append({"type": "text", "text": f"Frame {i + 1} — {label}:"})
            content.append({"type": "image_url", "image_url": {"url": _encode_image(frames[key])}})

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8000",
        "X-Title": "SimRobotics",
    }
    payload = {
        "model": OPENROUTER_MODEL,
        "response_format": {"type": "json_object"},
        "messages": [{"role": "user", "content": content}],
    }

    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
        data = json.loads(response.json()["choices"][0]["message"]["content"])
        if data.get("failure_type") not in FAILURE_TAXONOMY:
            data["failure_type"] = "abort_unrecoverable"
        if data.get("failed_checkpoint") not in GRASP_CHECKPOINTS:
            data["failed_checkpoint"] = "unknown"
        data["prompt_text"] = prompt  # expose for run-log saving
        return data
    except Exception as e:
        print(f"Error querying Gemini: {e}")
        try:
            print(response.text)
        except Exception:
            pass
        return None
