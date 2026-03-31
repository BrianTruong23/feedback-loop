Explainable Failure Reasoning for Language-Guided Robot Manipulation:
A Simulation-First Feedback Loop for Improved Task Success
Thang Truong
Auburn University
Abstract
This project studies whether a lightweight failure explanation module, used inside a feedback loop, can improve task success in language-guided robot manipulation. To keep the scope realistic for a class project and practical for development on a MacBook Pro, the system will be built and evaluated primarily in simulation. The core task is a constrained pick-and-place problem in clutter, where the agent receives a short language instruction such as “pick the red mug on the left” and attempts to select and grasp the correct object. When a failure occurs, the system will generate a structured failure reason and use that explanation to trigger a simple corrective action such as retrying, adjusting the grasp, or reselecting the target. The main evaluation will compare a baseline policy against an explanation-guided feedback policy in simulation using task success, recovery success, wrong-object rate, and latency overhead. A real robotic arm will be used only as a final small-scale validation step if the simulation pipeline is stable and time permits. The goal is not full-scale robot learning, but a prototype feedback loop with an initial empirical evaluation.
Index Terms— robot manipulation, vision-language models, failure explanation, simulation, feedback loop
I. Introduction
Language-guided robot manipulation is a useful setting for studying embodied AI because it requires a system to connect natural-language instructions with visual perception and action. However, even simple pick-and-place tasks can fail in cluttered scenes. A robot may choose the wrong object, fail to grasp the object securely, or mis-handle spatial language. These failures reduce task success and make robotic systems harder to trust or debug.
This project focuses on whether failure explanations can be operationally useful rather than only interpretable to humans. The idea is to detect a failure, generate a short explanation of what went wrong, and feed that explanation into a simple correction policy. For example, if the system explains that it picked a distractor because the target was partially occluded, the feedback policy may retry with a different target selection or grasp strategy.
To keep the project feasible, the work will be simulation-first. The full pipeline will be built, debugged, and evaluated in simulation on a local Mac-based setup. Only after the simulation results are stable will the system optionally be tested on a real robotic arm in a limited number of trials. This reduces hardware dependence and allows the main contribution to remain the explanation-feedback loop itself rather than hardware integration.
II. Problem Statement
Research question.
Can a lightweight explanation and feedback loop improve task success and recovery performance in a constrained language-guided manipulation task?
Scope and setting.
The main environment will be simulation. The task will be a tabletop pick-and-place scenario with a small number of household-like objects in clutter. Each episode will include:
a rendered scene,
a short language instruction describing the target object,
a baseline action attempt,
a success or failure outcome,
an optional retry based on explanation-driven feedback.
The language space will be intentionally limited to a small set of attributes and relations, such as color, object type, and simple spatial phrases. This restriction keeps the task measurable and realistic for the project timeline.
Core idea.
The baseline system will attempt to map language and visual input to a pick action. If the attempt fails, a failure reasoning module will produce:
a failure category, and
a short natural-language explanation.
A simple feedback policy will then map that explanation to one corrective action. Examples include:
retry the same pick,
reselect a different target object,
adjust grasp parameters,
terminate if confidence is too low.
The focus is not on building a large robotics model from scratch. Instead, the focus is on whether even simple explanations can help a baseline policy recover more effectively.
Expected outcome.
The expected result is a modest improvement in final task success and recovery success relative to the baseline, with some extra latency due to explanation generation. The project will explicitly report that tradeoff.
III. Proposed System
The system has four components:
A. Baseline Manipulation Policy
A lightweight baseline policy will perform target selection and grasp execution in simulation. This may use an existing open-source manipulation pipeline, a simple detector-plus-grasp heuristic, or a lightweight model that is practical to run and debug locally.
B. Failure Detection
A failed trial will be identified using simple outcome checks such as:
wrong object selected,
grasp failed,
object dropped,
timeout or no successful completion within a fixed attempt budget.
C. Failure Explanation Module
For each failed trial, the system will produce:
a structured failure type label, and
a short explanation.
Initial failure categories may include:
wrong-object selection,
target occluded,
grasp instability,
spatial misunderstanding,
no object reached.
To keep the scope manageable, the first version of the explanation module will be template-based or classifier-assisted rather than fully generative.
D. Feedback Policy
The explanation will be converted into a simple corrective action by a rule-based feedback policy. For example:
wrong-object selection → reselect target,
target occluded → retry with alternative object ranking,
grasp instability → adjust grasp and retry,
timeout → terminate.
This design keeps the project centered on operational usefulness rather than language generation quality alone.
IV. Data and Experimental Setup
Stage A: Simulation Episodes
The main dataset will be generated in simulation. Each episode will contain:
image or state observation,
language instruction,
predicted target/action,
outcome label,
timing information,
retry history if applicable.
The simulation scenes will be intentionally small and controlled. Object sets, clutter patterns, and language templates will be limited so that experiments are reproducible and easier to analyze.
Stage B: Failure Reasoning Dataset
A failure dataset will be built from unsuccessful simulation episodes. Each failure case will include:
scene snapshot or episode log,
failure type,
short explanation,
optional corrective action suggestion.
These explanations will initially be template-based and grounded in known simulator outcomes.
Stage C: Optional Real-Arm Validation
If the simulation pipeline is stable and time permits, a small number of final trials will be run on a real robotic arm. This stage is not the main evaluation. Its role is only to test whether the simulation-developed feedback logic transfers at a small scale. If hardware access is limited or integration becomes too time-consuming, this stage will be omitted without affecting the core contribution of the project.
V. Evaluation
Experimental Conditions
The project will compare:
Baseline manipulation
Explanation only (generate explanations but do not use them for correction)
Explanation + feedback
If time permits, an ablation may be added where feedback uses fixed heuristic reasons instead of generated or predicted explanations.
Quantitative Metrics
The following metrics will be reported:
Task success rate: percentage of trials where the correct target is picked successfully
Wrong-object rate: percentage of trials where a distractor is selected
Grasp success rate: percentage of trials where the grasp succeeds physically
Recovery success rate: percentage of failed first attempts corrected by feedback
Attempts to success: number of attempts needed to complete the task
Latency / time to completion: total runtime including explanation overhead
Qualitative Analysis
The project will present several representative cases:
successful correction after explanation,
failure cases where feedback did not help,
examples showing which explanation categories were most useful.
Retry Budget
The main setting will use at most two attempts total:
one initial attempt,
one feedback-driven retry.
This keeps the study focused and easy to analyze.
VI. Why Simulation-First Is the Right Scope
A simulation-first design is more realistic for this project for three reasons.
First, it reduces hardware dependence and makes development faster and more reproducible.
Second, it allows the main contribution to be the explanation-feedback loop instead of robotic-arm setup and debugging.
Third, it fits better with local development on a MacBook Pro, where simulation, logging, dataset construction, and smaller-scale model experiments are much easier to manage than a full real-robot pipeline. Your current setup is a 14-inch MacBook Pro with an M5 Pro chip and 48 GB of memory, which is a strong fit for local simulation, logging, and prototype development.
VII. Timeline
Week 1:
Finalize the simulator setup, define the constrained task, and implement the baseline manipulation pipeline.
Week 2:
Build logging, failure detection, and the first version of the failure explanation module.
Week 3:
Integrate the explanation-feedback loop and run the main simulation experiments.
Week 4:
Analyze results, create tables and figures, and write the final report.
If everything is working well, run a small optional real-arm validation at the end.
VIII. Risks and Fallback Plan
The main risk is that full feedback integration may take longer than expected. If that happens, the fallback plan is:
complete the baseline simulation pipeline,
build and evaluate the failure explanation module,
perform offline analysis showing how explanation-guided corrections would change outcomes,
report ablations and qualitative examples.
This still preserves the main research contribution while avoiding over-commitment to hardware integration.
IX. Conclusion
This project proposes a realistic, simulation-first study of explainable failure reasoning for language-guided robot manipulation. The goal is to test whether simple failure explanations can improve recovery and final success in a constrained pick-and-place task. By focusing on simulation as the primary environment and treating real-arm testing as an optional final validation step, the project becomes more feasible, reproducible, and better aligned with the available development setup. This scope keeps the work ambitious enough to be interesting, while still practical enough to finish well.