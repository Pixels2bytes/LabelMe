def test_llm_schema():
    response_schema = {
        "type": "object",
        "properties": {
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
                        "box_width": {"type": "number"},
                        "box_height": {"type": "number"},
                        "confidence": {"type": "number"}
                    },
                    "required": ["label", "bbox", "box_width", "box_height", "confidence"]
                }
            }
        },
        "required": ["detections"]
    }
    return response_schema