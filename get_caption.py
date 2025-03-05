from openai import OpenAI
import base64
import yaml
import os

# TODO: Config env variable -> export OPENAI_API_KEY=
client = OpenAI()

def encode_image(image_path):
    """ Translate image to base64 """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def analyze_image(image_path):
    base64_image = encode_image(image_path)

    response = client.chat.completions.create(
        model="gpt-4.5-preview",
        messages=[
            {"role": "system", "content": "You are an Mural expert to analyze damage region of the mural painting"},
            {"role": "user", "content": [
                {"type": "text", "text": "Analyze this image damage region and describe its contents:"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
            ]}
        ],
    )

    return response.choices[0].message.content

config_path = "checkpoints/config.yml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

flist_path = config.get("TRAIN_OVERLAY_FLIST")

with open(flist_path, "r") as f:
    image_paths = [line.strip() for line in f.readlines()]

caption_save_path = "checkpoints/captions.yml"
if os.path.exists(caption_save_path):
    with open(caption_save_path, "r") as f:
        captions = yaml.safe_load(f) or {}
else:
    captions = {}

for image_path in image_paths:
    if image_path in captions:
        print(f"Skipping {image_path}, already processed.")
        continue

    result = analyze_image(image_path)
    captions[image_path] = result

    with open(caption_save_path, "w") as f:
        yaml.dump(captions, f, default_flow_style=False)

    print(f"Saved caption for {image_path}")

