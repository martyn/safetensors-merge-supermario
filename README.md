# safetensors-merge-supermario

Combine any two models using a Super Mario merge as described in the linked whitepaper.

## About

Combine capabilities from multiple models. Built for Stable Diffusion XL / Stable Diffusion XL Turbo but it matches any safetensor keys with the same name.

### Example

| **Model** | **Description** | **Image**(same seed) |
|-----------|-----------------|-----------|
| **sd_xl_turbo** | Attempting 1024 | <img src="assets/before_xl_turbo.png" alt="SDXL turbo attempting to render at 1024" width="256" height="256"> |
| **sdxl base 1.0** | Attempting to use SDTurboScheduler | <img src="assets/before_xl.png" alt="SDXL attempting to use SDTurboScheduler" width="256" height="256"> |
| **merged** | Mario merged | <img src="assets/after.png" alt="Merged model successfully rendering 1024" width="256" height="256"> |

## Usage

```
python3 merge.py -p 0.13 -lambda 3.0 sdxl_base.safetensors sd_xl_turbo_1.0_fp16.safetensors sdxl_merged.safetensors
```

Note: This also works with the arguments reversed.

## Models:

* https://huggingface.co/martyn/sdxl-turbo-mario-merge - SD Turbo XL merged with SDXL Base

## References:

* https://arxiv.org/pdf/2311.03099.pdf - Language Models are Super Mario: Absorbing Abilities from Homologous Models as a Free Lunch
* https://stability.ai/research/adversarial-diffusion-distillation - SXDL Turbo - Adversarial Diffusion Distillation
