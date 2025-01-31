from typing import Dict, Any, List

# ------------------------------------------------------------------------------------
#                               1. INPUT SCHEMA
# ------------------------------------------------------------------------------------
# We define a single schema that covers the entire pipeline's inputs.
# For each field:
#    - "type": The Python type we expect (e.g., str, int, list, float).
#    - "required": Whether this field must be provided.
#    - "default": A default value if not provided.
#    - "constraints": A lambda or function to validate further if needed.
INPUT_SCHEMA = {
    "story_script": {
        "type": str,
        "required": True,
    },
    "lang_code": {
        "type": str,
        "required": False,
        "default": "a"
    },
    "voice": {
        "type": str,
        "required": False,
        "default": "af_bella"
    },
    "image_prompts": {
        "type": list,
        "required": True,
        # example constraints if needed: "constraints": lambda x: len(x) > 0
    },
    "video_resolution": {
        "type": list,
        "required": False,
        "default": [1280, 720],
        # If needed, you can add a constraints check for len=2
    },
    "image_resolution": {
        "type": list,
        "required": False,
        "default": [1344, 768],
        # If needed, you can add a constraints check for len=2
    },
    "job_id": {
        "type": str,
        "required": False,
        "default": None
    },
    "output_video_name": {
        "type": str,
        "required": False,
        "default": "final_video_with_subtitles.mp4"
    },
    "video_fps": {
        "type": int,
        "required": False,
        "default": 30
    },

    # -------------- SDXL Generation Params --------------
    'sdxl_negative_prompt': {
        'type': str,
        'required': False,
        'default': None
    },
    "scheduler": {
        "type": str,
        "required": False,
        "default": "DPMSolverMultistep"
    },
    "num_inference_steps_base": {
        "type": int,
        "required": False,
        "default": 40
    },
    "num_inference_steps_refiner": {
        "type": int,
        "required": False,
        "default": 60
    },
    "guidance_scale": {
        "type": float,
        "required": False,
        "default": 7.5
    },
    "seed": {
        "type": int,
        "required": False,
        "default": None
    },
    'strength': {
        'type': float,
        'required': False,
        'default': 0.3
    },
    'high_noise_frac': {
        'type': float,
        'required': False,
        'default': None
    },
}

# ------------------------------------------------------------------------------------
#                               2. VALIDATION
# ------------------------------------------------------------------------------------
def ValidateUserInput(data: Dict[str, Any], schema: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Validates the user input dictionary against INPUT_SCHEMA.
    Returns a dictionary with validated values (including defaults).
    Raises ValueError if any required field is missing or type mismatch occurs.
    """
    validated = {}
    errors = []

    for field, rules in schema.items():
        # If user provided the field
        if field in data:
            val = data[field]
            # Check type
            expected_type = rules.get("type")
            if expected_type and not isinstance(val, expected_type):
                errors.append(f"Field '{field}' must be of type {expected_type.__name__}, got {type(val).__name__}")
                continue

            # Check constraints if present
            if "constraints" in rules and not rules["constraints"](val):
                errors.append(f"Field '{field}' does not satisfy constraints.")
                continue

            validated[field] = val
        else:
            # Not provided by user, check if required
            if rules.get("required", False):
                errors.append(f"Missing required field '{field}'")
            else:
                # Use default if available
                validated[field] = rules.get("default", None)

    if errors:
        raise ValueError("Validation errors:\n" + "\n".join(errors))

    return validated