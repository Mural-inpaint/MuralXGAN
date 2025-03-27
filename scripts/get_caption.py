from openai import OpenAI
import base64
import yaml
import os
import json

# TODO: Config env variable -> export OPENAI_API_KEY=
client = OpenAI()

def encode_image(image_path):
    """ Translate image to base64 """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def analyze_image(image_path):
    base64_image = encode_image(image_path)

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an Mural expert to analyze damage region of the mural painting"},
            {"role": "user", "content": [
                {"type": "text", "text": "Analyze this image damage region and describe its contents: color (red is for highlighting damage region, so ignore red.), damage region, damage type, severity, and position. Construct it in json format { \"damage_type\": \"shedding\", # options: [\"shedding\", \"fading\", \"discoloration\"] \"damage_severity\": \"moderate\", # options: [\"mild\", \"moderate\", \"severe\"] \"colors_affected\": [\"green\", \"yellow\"], \"position\": \"upper right\", \"description\": \"The shedding is concentrated on the upper right corner, affecting the deity's robe.\"}"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
            ]}
        ],
        response_format={"type": "json_object"}
    )

    response_json = json.loads(response.choices[0].message.content)
    return validate_response(response_json, image_path)


def validate_response(data, image_path):
    """Ensure all required fields are present, assign defaults if missing"""
    picture_name = os.path.basename(image_path)
    required_fields = {
        "picture_name": picture_name,
        "damage_type": "unknown",
        "damage_severity": "unknown",
        "colors_affected": [],
        "position": "unknown",
        "description": "No description available."
    }

    # Fill missing fields with default values
    for key, default in required_fields.items():
        if key not in data or not data[key]:
            data[key] = default

    # Ensure colors_affected is a list
    if not isinstance(data["colors_affected"], list):
        data["colors_affected"] = [data["colors_affected"]] if data["colors_affected"] else []

    # Generate full_text for embeddings
    data["full_text"] = (
        f"Damage type: {data['damage_type']}. Severity: {data['damage_severity']}. "
        f"Affected colors: {', '.join(data['colors_affected'])}. {data['description']}"
    )

    return data


config_path = "../checkpoints/config.yml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

flist_path = config.get("TEST_OVERLAY_FLIST")
with open(flist_path, "r") as f:
    image_paths = [line.strip() for line in f.readlines()]

caption_save_path = "../checkpoints/test/test-captions.yml"
if os.path.exists(caption_save_path):
    with open(caption_save_path, "r") as f:
        captions = yaml.safe_load(f) or {}
else:
    captions = {}

json_save_path = "../checkpoints/test/test-captions.json"
if os.path.exists(json_save_path):
    with open(json_save_path, "r") as f:
        try:
            json_captions = json.load(f)
        except json.JSONDecodeError:
            json_captions = {}
else:
    json_captions = {}

for image_path in image_paths:
    picture_name = os.path.basename(image_path)

    if picture_name in captions:
        print(f"Skipping {image_path}, already processed.")
        continue

    result = analyze_image(image_path)
    captions[picture_name] = result["full_text"]
    json_captions[picture_name] = result

    with open(caption_save_path, "w") as f:
        yaml.dump(captions, f, default_flow_style=False)

    with open(json_save_path, "w") as f:
        json.dump(json_captions, f, indent=4)

    print(f"Saved caption for {image_path}")

