from openai import OpenAI
import base64

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

    return response.choices[0]

image_path = "checkpoints/test.png"
result = analyze_image(image_path)
print(result)

