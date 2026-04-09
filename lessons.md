# Lessons Learned: Moving To A Checkpoint-First Recovery Architecture

This document explains the recent architectural change in this project: moving away from a single-shot "VLM explains failure and predicts a new pixel target" pipeline toward a more structured "checkpoint-first failure classification + recovery primitive" pipeline.

The goal of this change is not to make the system more complicated for its own sake. The goal is to make robotic recovery more reliable, easier to debug, and easier to evaluate.

## 1. The Old Architecture

The older version of the system followed this pattern:

1. Detect the target object with OWL-ViT.
2. Convert the detected 2D image center into a 3D grasp target.
3. Attempt the grasp.
4. If the grasp failed, send a single image to the VLM.
5. Ask the VLM to:
   - explain why the failure happened
   - predict corrected pixel coordinates
   - optionally suggest a yaw correction
6. Convert those corrected pixels into a new 3D grasp point.
7. Retry the grasp.

This was a reasonable first design, but it had several weaknesses:

- The VLM had to do too many things at once.
- A single post-failure image often did not contain enough information to explain the failure correctly.
- Free-form pixel correction was fragile.
- Small image-space mistakes could turn into large 3D control mistakes.
- Recovery behavior was hard to reason about because the VLM was effectively inventing the next action.

In short: the system could produce plausible explanations, but the recovery action was not reliably grounded.

## 2. The New Architecture

The updated design separates diagnosis from recovery.

The new flow is:

1. Detect the object with OWL-ViT.
2. Attempt the grasp.
3. Capture a short sequence of front-view frames across the grasp attempt.
4. Ask the VLM to classify the failure using those frames.
5. Map the classified failure to a predefined recovery primitive.
6. Retry using that primitive.

The key change is this:

The VLM is no longer responsible for directly inventing the exact next grasp target in pixel space.

Instead, the VLM acts as a structured failure analyst. The control code remains responsible for deciding how to recover.

## 3. Checkpoint-First Reasoning

Checkpoint-first reasoning means the system analyzes the grasp as a sequence of simple stages instead of asking one broad question like "why did the grasp fail?"

The grasp is broken into checkpoints such as:

- target identified
- gripper aligned above target
- fingers enclosing target
- object lifted clear of bin
- correct object in gripper

The VLM is asked to determine the first checkpoint that failed.

Why this helps:

- It reduces ambiguity.
- It prevents the model from mixing up root cause and downstream symptom.
- It produces failure explanations that are more actionable.

Example:

- If the robot never reached the object, the problem is not slip.
- If the robot clearly touched the object and only lost it during lift, the problem is not initial localization.

This makes recovery selection more sensible.

## 4. Temporal Evidence Instead Of Single-Frame Reasoning

The new system uses a short ordered sequence of frames instead of just one image.

Typical evidence frames include:

- pre-hover
- contact
- post-close
- post-lift
- retracted clean view

This is important because grasp failures are dynamic.

A single image often cannot tell the difference between:

- missing the object entirely
- touching the object off-center
- grasping correctly but slipping during lift
- selecting the wrong object

A sequence gives the VLM causal context across time.

## 5. Fixed Failure Taxonomy

The new architecture uses a fixed list of allowed failure types rather than open-ended descriptions.

Current categories include:

- wrong_object
- no_object_reached
- grasp_pose_bad
- slip_after_contact
- target_occluded
- needs_yaw_adjustment
- depth_plane_bad
- abort_unrecoverable

Why this matters:

- Each label has a specific meaning.
- Each label can map to a known recovery strategy.
- Logs and evaluation become easier to compare across runs.
- The VLM is constrained to choose among known explanations instead of inventing arbitrary categories.

This is similar in spirit to AHA, where failure reasoning is tied to a known manipulation failure taxonomy.

## 6. Recovery Primitives

A recovery primitive is a predefined retry strategy chosen by code.

Examples:

- re-run OWL-ViT detection
- re-detect with a more specific prompt
- rotate the gripper before descending
- lower the grasp depth slightly
- increase settling time during grip closure
- abort if the scene is too disrupted

This is different from free-form pixel correction.

Under the older design, the VLM might say something like:

- move 23 pixels right
- move 11 pixels down
- rotate 18 degrees

That sounds precise, but it is often brittle because:

- image coordinates are noisy
- projection from image space to world space is imperfect
- the VLM is not a geometric planner

With recovery primitives, the VLM does something narrower and more reliable:

- classify the failure
- let code choose the corresponding repair policy

This keeps the model in a high-level reasoning role and keeps the robot controller in charge of motion decisions.

## 7. Diagnosis And Correction Are Now Separate

This is one of the most important conceptual changes.

Old idea:

- Ask the VLM to diagnose the problem and directly prescribe the corrected grasp target.

New idea:

- Ask the VLM to diagnose the problem only.
- Use code to translate that diagnosis into a recovery strategy.

Why this is better:

- Easier to test
- Easier to debug
- Easier to compare policies
- Safer for robot behavior

If recovery fails, we can now ask:

- Was the classification wrong?
- Was the chosen primitive wrong?
- Was the primitive correct, but the execution poor?

That was much harder to answer in the old system.

## 8. What Improved

The architecture is now better in several ways:

- Failure reasoning is more structured.
- The VLM sees temporal context instead of one static image.
- Recovery is policy-based instead of free-form.
- Logging is more interpretable.
- The system is closer to a robotics debugging tool than a one-shot image chatbot.

## 9. What Still Needs Improvement

This redesign is a major step forward, but it does not solve everything.

Open issues remain:

- Some recovery branches still need better target-state tracking across retries.
- Occlusion recovery should use a genuine viewpoint change, not just same-view re-detection.
- Recovery policies are still fairly coarse.
- Evaluation should separate:
  - checkpoint classification accuracy
  - failure-type classification accuracy
  - recovery-policy selection accuracy
  - end-to-end task success

In other words, the new design is better structured, but it still needs stronger recovery execution and better evaluation.

## 10. Core Takeaway

The main lesson is:

Natural-language failure reasoning is useful, but only when it is tightly constrained and connected to concrete recovery behavior.

The VLM should not be treated as a full geometric control policy.

The VLM is most useful here as:

- a failure classifier
- a temporal visual analyst
- a selector over known recovery strategies

That is the architectural direction this project now follows.
