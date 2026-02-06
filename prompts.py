# prompts.py

prompt_target = (
    "The location shown in the image is the target you need to guide the drone to find within the city. "
    "The target could be an object or a building. Please identify what this target is and provide a detailed "
    "description of the target, while also describing the environment in which the target is located."
    "Finally, present the target and the surrounding environmental elements in a format like: ([sidewalk], [building], [trees])"
)

prompt_judge = (
    "Please determine whether the search target you are searching for exists in this picture. "
    "Please check to ensure that the search target must exist before judging the task success. "
    "If the search target is found, please plus 'Task successful' in the end of the reply."
)

answer_target = (
    "The search target is a large sign that reads 'Cheesspod‘. The sign is prominently displayed on the side of the building, "
    "which appears to be a modern structure with a white facade and blue-tinted windows. "
    "There are also air conditioning units mounted on the exterior wall above the sign."
)

prompt_rel = (
    "What obvious objects (at most ten and at least three) are contained in the image that can help me locate the position from a distance? "
    "Please only return objects, separated by periods (.) in the format. Using lowercase letters"
)
