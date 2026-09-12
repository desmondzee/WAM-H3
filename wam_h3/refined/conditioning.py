def prompt_variants(task):
    task = task.strip().rstrip(".")
    minimal = f"The robot arm uses its gripper to {task}."
    cameras = ("A synchronized split-screen recording of the same action from two cameras: "
               "a fixed external view on the left and a wrist-mounted camera on the right. "
               "The right camera is rigidly attached to the robot wrist and moves with the arm. ")
    aware = cameras + minimal + " Both views show the same continuous action at the same time, with physically consistent object motion."
    return {"minimal": minimal, "camera": aware,
            "constraints": aware + " No delay, static wrist view, or teleportation.",
            "physical": cameras + f"Using physical contact with its gripper, the robot performs the task: {task}."}
