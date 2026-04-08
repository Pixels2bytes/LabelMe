def test_llm_schema():
    response_schema = {
        "type": "object",
        "properties": {
            "image_width": {"type": "integer"},
            "image_height": {"type": "integer"},
            "detections": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "label": {"type": "string"},
                        "bbox": {
                            "type": "array",
                            "items": {"type": "number"},
                            "minItems": 4,
                            "maxItems": 4
                        },
                        "width": {"type": "number"},
                        "height": {"type": "number"},
                        "confidence": {"type": "number"}
                    },
                    "required": ["label", "bbox", "width", "height", "confidence"]
                }
            }
        },
        "required": ["image_width", "image_height", "detections"]
    }
    return response_schema