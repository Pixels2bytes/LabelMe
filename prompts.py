def img_ground_truth_prompt(classes: list[str]):
    prompt = f"""
    You are a precise computer vision assistant for generating object detection annotations.

    Given an image, detect all instances of the following classes:
    {', '.join(classes)}

    Instructions:
    1. Detect all objects belonging to the listed classes.
    2. Draw tight bounding boxes around each object.
    3. Use pixel coordinates (NOT normalized).
    4. Coordinates must follow this exact format:
    [x_min, y_min, x_max, y_max]
    5. Provide bounding box width and height.
    6. Provide image width and height.
    7. Assign a confidence score between 0.0 and 1.0.

    Return ONLY valid JSON. No explanations. No code fences.

    Required JSON format:
    {{
        "image_width": <int>,
        "image_height": <int>,
        "detections": [
            {{
                "label": "<label>",
                "bbox": [x_min, y_min, x_max, y_max],
                "width": <int>,
                "height": <int>,
                "confidence": <float>
            }}
        ]
    }}

    Rules:
    - label must be one of: {', '.join(classes)}
    - All coordinates must be integers
    - width = x_max - x_min
    - height = y_max - y_min
    - Ensure bounding boxes are accurate and tight
    """
    return prompt