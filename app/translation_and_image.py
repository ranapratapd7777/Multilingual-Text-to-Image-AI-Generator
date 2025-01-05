from googletrans import Translator
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

def get_translation(text, dest_lang):
    translator = Translator()
    translated_text = translator.translate(text, dest=dest_lang)
    return translated_text.text


class CFG:
    device = "cuda" if torch.cuda.is_available() else "cpu"  # Dynamically set device based on availability
    seed = 42
    generator = torch.Generator("cpu").manual_seed(seed)  # Use CPU here
    image_gen_steps = 35
    image_gen_model_id = "stabilityai/stable-diffusion-2"
    image_gen_size = (900, 900)
    image_gen_guidance_scale = 9
    prompt_gen_model_id = "gpt3"
    prompt_dataset_size = 6
    prompt_max_length = 12

# Load and move model to the device
image_gen_model = StableDiffusionPipeline.from_pretrained(
    CFG.image_gen_model_id, torch_dtype=torch.float16,
    revision="fp16", use_auth_token='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', guidance_scale=9
)
image_gen_model = image_gen_model.to(CFG.device)


def generate_image(prompt):
    image = image_gen_model(
        prompt, num_inference_steps=CFG.image_gen_steps,
        generator=CFG.generator,
        guidance_scale=CFG.image_gen_guidance_scale
    ).images[0]

    image = image.resize(CFG.image_gen_size)
    return image
