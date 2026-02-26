import random
import os
import shutil
from dotenv import load_dotenv
from click import prompt
from google import genai
from anthropic import Anthropic
from openai import OpenAI
from google.genai import types
import base64

load_dotenv(dotenv_path="utils/.env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPEN_MODEL = "gpt-4-0613"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-2.5-flash-image"

CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY", "")
CLAUDE_MODEL = "claude-2"

OUTPUT_DIR = "resources/generated_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)
IMAGE_SIZE = "512x512"
IMAGE_FORMAT = "PNG"
GEN_NUMBER = 10

client = genai.Client(api_key=GEMINI_API_KEY)

def prompt_randomizer():
    focuses = ["human", "young human", "animal", "young boy", "female", "young girl", "male", "child", "teenager", "adult", "elderly person"]
    focus = random.choice(focuses)

    blank_bkg = """
    The image should have a blank background, such as a solid white color or a simple gradient, to ensure that the focus is on the subject and their activity. The background should not contain any distracting elements or details that could take away from the main subject of the image.
    """
    match_bkg = f"""The image should have a background that matches the activity being performed by the main subject. For example,
    if the {focus} is cooking, the background could be a kitchen setting. If the {focus} is exercising, the background could be a gym or outdoor setting. The background should complement the activity and enhance the overall realism of the image without overpowering the main subject.
    """
    bkg_options = [blank_bkg, match_bkg]
    bkg = random.choice(bkg_options)

    prompt = f"""
    Generate an image of a {focus} doing a mundane task or exciting activity, such as cooking, cleaning, exercising, 
    or playing a sport. The image should be realistic and high to medium quality. The image should be in {IMAGE_FORMAT} format
    and should be no larger than {IMAGE_SIZE}. {bkg}
    """
    return prompt, focus


def gem_generate_image():
    prompt, focus = prompt_randomizer()
    filename = f"{focus}_{random.randint(1000, 9999)}_{IMAGE_SIZE}_{random.randint(1000, 9999)}.{IMAGE_FORMAT.lower()}"
    filepath = os.path.join(OUTPUT_DIR, filename)
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt
    )

    for part in response.candidates[0].content.parts:
        if part.inline_data:
            with open(filepath, "wb") as f:
                f.write(part.inline_data.data)

    print(f"Saved as {filename}")


def main():
    for count in range(GEN_NUMBER):
        print(f"Generating image {count + 1}/{GEN_NUMBER}...")
        gem_generate_image()


if __name__ == "__main__":
    main()