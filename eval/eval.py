
from functools import partial
from pydantic import BaseModel
import base64
from openai import OpenAI
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from io import BytesIO
from PIL import Image
from torchmetrics.functional.multimodal.clip_score import clip_score
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-base-patch16")
MODEL_NAME = "gpt-4o"
def calculate_clip_score(images, prompts):
    modified_prompts = [p.replace("<phigep>", "young male person with brown hair") for p in prompts]
    images_int = (images * 255).astype("uint8")
    clip_score = clip_score_fn(torch.from_numpy(images_int).permute(0, 3, 1, 2), modified_prompts).detach()
    return round(float(clip_score), 4)


def prepare_image_for_clip_cv2(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=0)
    return image


def optimized_canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    median_intensity = np.median(gray)
    lower_threshold = int(0.33 * median_intensity)
    upper_threshold = int(1.33 * median_intensity)
    edges = cv2.Canny(gray, lower_threshold, upper_threshold)
    return edges

def plot_pixel_histogram(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    histogram, bin_edges = np.histogram(gray, bins=256, range=(0, 255))
    plt.figure(figsize=(8, 4))
    plt.plot(bin_edges[:-1], histogram, lw=2, color='black')
    plt.title("Pixel Intensity Histogram")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.show()

client = OpenAI(api_key="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")

class EvalScore(BaseModel):
    score:float
    textual_justification:str


def encode_and_prepare_image(image_path):
    base64_image = encode_image(image_path)
    return base64_image

# Function to encode the image, openai want jpeg encoded as base64 it seems
def encode_image(image_path):
    image = Image.open(image_path)
    image = image.convert("RGB")
    buffer = BytesIO()
    image.save(buffer, format="JPEG")  # covnert and reload
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def decode_base64_image(base64_string):
    if base64_string.startswith("data:image"):
        # Split on the first comma
        base64_string = base64_string.split(",", 1)[1]

    image_data = base64.b64decode(base64_string)
    pil_img = Image.open(BytesIO(image_data))
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def optimized_canny(base64_string):
    raw_data = base64.b64decode(base64_string)
    pil_original = Image.open(BytesIO(raw_data))
    cv_original = cv2.cvtColor(np.array(pil_original), cv2.COLOR_RGB2BGR)


    gray = cv2.cvtColor(cv_original, cv2.COLOR_BGR2GRAY)
    median_intensity = np.median(gray)
    lower_threshold = int(0.33 * median_intensity)
    upper_threshold = int(1.33 * median_intensity)
    edges = cv2.Canny(gray, lower_threshold, upper_threshold)
    edges_pil = Image.fromarray(edges).convert("RGB")

    buffer = BytesIO()
    edges_pil.save(buffer, format="JPEG")  
    edges_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    #print(edges_base64)
    return edges_base64

def construct_message_for_image(base64_encoded_image):
    return {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_encoded_image}"},
                },

def construct_user_text_message(prompt_str):
    return {
                    "type": "text",
                    "text": prompt_str,
                },

def eval_person_llm(generated_image):
    completion=client.beta.chat.completions.parse(
    model=MODEL_NAME,
    messages=[
        {
            "role": "user",
            "content": [
                *construct_user_text_message("Is the image depicting a person or multiple people? Rate with a score of 0 (not a person) or 1 (a person).  "),
                *construct_message_for_image(generated_image),
            ],
        }
    ],
    response_format=EvalScore,
)
    return completion.choices[0].message.parsed


def eval_same_person_llm(base_image, generated_image):
    completion=client.beta.chat.completions.parse(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": [
                    *construct_user_text_message("Do both image contain the exact same person? Rate based on general facial features, as well as Hair Color,Hair Length, Eyecolor, Beard etc. with a score from 0 to 1. 1 Would mean its the exact same person."),
                    *construct_message_for_image(base_image),
                    *construct_message_for_image(generated_image),
                ],
            }
        ],
            response_format=EvalScore,
    )
    return completion.choices[0].message.parsed





def eval_prompt_adherence_llm(base_image, generated_image, generation_prompt):
    completion=client.beta.chat.completions.parse(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": [
                    *construct_user_text_message(f"The first image contains a real image of <phigep> (custom token for the person). The second is a generated one based on the following prompt: \nPROMPT:{generation_prompt}\n Rate the Adherence to the prompt and give back a single score between 0 and 1. 1 would be perfect adherence to prompt. Include adherence to both the person (is the person similar to <phigep> as well as adherence to the scene description of the prompt.)"),
                    *construct_message_for_image(base_image),
                    *construct_message_for_image(generated_image),
                ],
            }
        ],
            response_format=EvalScore,
    )
    return completion.choices[0].message.parsed


def eval_prompt_adherence_llm_simpler_prompt(base_image, generated_image, generation_prompt):
    completion=client.beta.chat.completions.parse(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": [
                    *construct_user_text_message(f"The first image contains a real image of <phigep> (custom token for the person). The second is a generated one based on the following prompt: \nPROMPT:{generation_prompt}\n Rate the Adherence to the prompt and give back a single score between 0 and 1. 1 would be perfect adherence to prompt."),
                    *construct_message_for_image(base_image),
                    *construct_message_for_image(generated_image),
                ],
            }
        ],
            response_format=EvalScore,
    )
    return completion.choices[0].message.parsed

def eval_image_quality_tournament(image1, image2):
    completion=client.beta.chat.completions.parse(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": [
                    *construct_user_text_message(f"Evaluate both attached Images based on Image Fidelity, Image Resolution overall Quality and Contrast. Give it a score from 0 to 1, where 0 means you totally prefer the first image and 1 means the second image is of way higher quality."),
                    *construct_message_for_image(image1),
                    *construct_message_for_image(image2),
                ],
            }
        ],
            response_format=EvalScore,
    )
    return completion.choices[0].message.parsed



def eval_image_quality_llm(generated_image):
    completion=client.beta.chat.completions.parse(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": [
                    *construct_user_text_message("The following Image is ai generated, the second is the canny edge result of it. Evaluate its image quality visually. Just the quality, not the artifacts. Rate with a score from 0 to 1. "),
                    *construct_message_for_image(generated_image),
                    *construct_message_for_image(optimized_canny(generated_image)),
                ],
            }
        ],
            response_format=EvalScore,
    )
    return completion.choices[0].message.parsed



def eval_limb_quality_llm(generated_image):
    completion=client.beta.chat.completions.parse(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": [
                    *construct_user_text_message("The following Image is ai generated, the second is the canny edge result of it. Evaluate the quality of generated hands, limbs and proportions in general in a score from 0 to 1. 1 would mean perfectly believable limbs and proportions"),
                    *construct_message_for_image(generated_image),
                    *construct_message_for_image(optimized_canny(generated_image)),
                ],
            }
        ],
            response_format=EvalScore,
    )
    return completion.choices[0].message.parsed

def eval_artifacts_quality_llm(generated_image):
    completion=client.beta.chat.completions.parse(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": [
                    *construct_user_text_message("The following Image is ai generated, the second is the canny edge result of it. Spot and evaluate andy artifacts and issues in light, shadows, facial features you find. Rate with a score from 0 to 1. Where 1 means no issues or artifacts at all."),
                    *construct_message_for_image(generated_image),
                    *construct_message_for_image(optimized_canny(generated_image)),
                ],
            }
        ],
            response_format=EvalScore,
    )
    return completion.choices[0].message.parsed

from typing import Optional, Dict, Any

import pandas as pd

def test_all_scores(
    foldername,
    single_param_funcs,
    two_param_funcs,
    three_param_funcs,
    base_image_path,
    prompt_dict
):
    base_image_b64 = encode_image(base_image_path)
    func_list = single_param_funcs + two_param_funcs + three_param_funcs
    func_names = [f.__name__ for f in func_list]
    rows = []
    from tqdm.notebook import tqdm
    for filename in tqdm(os.listdir(foldername), desc="Processing images"):
        if not filename.lower().endswith((".png", ".jpg",".jpeg")):
            continue
        parts = filename.split("_")
        if len(parts) < 4:
            continue
        loraname = parts[0]
        training_step = parts[1]
        prompt_id = parts[2]
        timestamp = parts[3].split(".")[0]
        gen_b64 = encode_and_prepare_image(os.path.join(foldername, filename))
        #print(gen_b64)
        generation_prompt = prompt_dict.get(prompt_id, "")
        scores = []
        clip_score = calculate_clip_score(prepare_image_for_clip_cv2(os.path.join(foldername, filename)),[generation_prompt])
        idx = 0
        for func in single_param_funcs:
            r = func(gen_b64)
            scores.append(r.score if hasattr(r, "score") else None)
            idx += 1
        for func in two_param_funcs:
            r = func(base_image_b64, gen_b64)
            scores.append(r.score if hasattr(r, "score") else None)
            idx += 1
        for func in three_param_funcs:
            r = func(base_image_b64, gen_b64, generation_prompt)
            scores.append(r.score if hasattr(r, "score") else None)
            idx += 1

        rows.append([loraname, training_step, prompt_id, timestamp] + scores + [clip_score])

    columns = ["loraname", "training_step", "prompt_id", "timestamp"] + func_names + ["clip_score"]
    return pd.DataFrame(rows, columns=columns)