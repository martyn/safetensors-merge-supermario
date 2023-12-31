# safetensors-merge-supermario

Combine any two models using a Super Mario merge(DARE) as described in the linked whitepaper.

## About

Combine capabilities from multiple models. Works with:

* Stable Diffusion (1.5, XL/XL Turbo)
* LLMs(Mistral, Llama, etc)
* LoRas(must be same size)
* Any two homologous models

### Example

| **Model** | **Description** | **Image**(same seed) |
|-----------|-----------------|-----------|
| **sd_xl_turbo** | Attempting 1024 | <img src="assets/before_xl_turbo.png" alt="SDXL turbo attempting to render at 1024" width="256" height="256"> |
| **sdxl base 1.0** | Attempting to use SDTurboScheduler | <img src="assets/before_xl.png" alt="SDXL attempting to use SDTurboScheduler" width="256" height="256"> |
| **merged** | Mario merged(DARE) | <img src="assets/after.png" alt="Merged model successfully rendering 1024" width="256" height="256"> |

## Usage

```
python3 merge.py -p [weight drop probability] -lambda [scaling factor] [base_safetensors_model_file_or_folder] [model_to_merge] [output_path]
```

### Example

```
python3 merge.py -p 0.13 -lambda 3.0 sdxl_base.safetensors sd_xl_turbo_1.0_fp16.safetensors sdxl_merged.safetensors
```

Note: This also works with arguments reversed.

## Models

* https://huggingface.co/martyn/sdxl-turbo-mario-merge - SD Turbo XL merged with SDXL Base
* https://civitai.com/models/215796 - Top Rated - TurboXL+LCM - Super Mario Merge
* https://huggingface.co/martyn/llama-megamerge-dare-13b - A llama 13b mega merge created using `hf_merge.py`
* https://huggingface.co/collections/martyn/dare-llm-merges-6581f52c0fb25b9aa26fb180 - A collection of LLM merges
* https://huggingface.co/collections/martyn/dare-diffusion-merges-6581f5fab1c4ab777ad43cf7 - A collection of text-to-image diffusion model merges

## ComfyUI workflow

* [ComfyUI merged base, turbo at 1024](assets/comfyui-sdxl-base-turbo-merged.json)

## Changelog

* Dec 27 2023: Add support for using files, folders, or hf repos in the hf_merge.py merge list.
* Dec 27 2023: Added mergekit-compatible yaml support for `hf_merge.py`. Always runs dare and ignores options outside model specification. weight is p and density is 1/λ.
* Dec 12 2023: Added `hf_merge.py` for merging hf repos.
* Dec 12 2023: Added support for folders. You can now merge LLMs(mistral, llama, etc) and other large models. Folders with `.bin` files are supported - the first specified model in the cli must be in `.safetensors` format.
* Nov 28 2023: Initial release supporting stable diffusion.

## References

* https://arxiv.org/pdf/2311.03099.pdf - Language Models are Super Mario: Absorbing Abilities from Homologous Models as a Free Lunch
* https://github.com/yule-BUAA/MergeLM - GitHub for the linked whitepaper
* https://stability.ai/research/adversarial-diffusion-distillation - SXDL Turbo - Adversarial Diffusion Distillation
