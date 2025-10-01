---
tags:
- text-to-image
- flux
- lora
- diffusers
- template:sd-lora
- fluxgym
widget:
-   output:
        url: sample/personal-lora-fluxdev-keyword_002400_00_20250106190431.png
    text: <phigep>
-   output:
        url: sample/personal-lora-fluxdev-keyword_002400_01_20250106190443.png
    text: <phigep> sitting on a bench in central park wearing a black hoodie and pink
        crocs
-   output:
        url: sample/personal-lora-fluxdev-keyword_002400_02_20250106190455.png
    text: <phigep> playing chess in space
base_model: black-forest-labs/FLUX.1-dev
instance_prompt: <phigep>
license: other
license_name: flux-1-dev-non-commercial-license
license_link: https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/LICENSE.md
---

# personal_lora_fluxdev_keyword

A Flux LoRA trained on a local computer with [Fluxgym](https://github.com/cocktailpeanut/fluxgym)

<Gallery />

## Trigger words

You should use `<phigep>` to trigger the image generation.

## Download model and use it with ComfyUI, AUTOMATIC1111, SD.Next, Invoke AI, Forge, etc.

Weights for this model are available in Safetensors format.

