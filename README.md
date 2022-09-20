# Digil Diffusion

*Digil Diffusion* is an amalgamation of various open source projects that contribute to the use of and extention to the utility of Stable Diffusion, **an AI driven machine learning model used to generate digital images from natural language descriptions**.


## Waifu Diffusion

Waifu Diffusion is the name for the project of finetuning Stable Diffusion on images and captions downloaded through Danbooru.

Model Link: https://huggingface.co/hakurei/waifu-diffusion  

[Training Guide](https://github.com/harubaru/waifu-diffusion/blob/main/docs/en/training/README.md)


## Web UI, GFPGAN, Real-ESRGAN, ESRGAN, and More

Has an implemetation of the [AUTOMATIC1111 Web UI](https://github.com/AUTOMATIC1111/stable-diffusion-webui) *(webui.cmd)*, to drive a number of settings and additional features (e.x. [GFPGAN](https://github.com/gSOLO/GFPGAN) for face fixing, and [Real-ESRGAN](https://github.com/gSOLO/Real-ESRGAN) for upscaling).

## Automatic Web UI Features

[<font size="12">Detailed feature showcase with images, art by Greg Rutkowski</font>](https://github.com/AUTOMATIC1111/stable-diffusion-webui-feature-showcase)

- Original txt2img and img2img modes
- One click install and run script (but you still must install python and git)
- Outpainting
- Inpainting
- Prompt matrix
- Stable Diffusion upscale
- Attention
- Loopback
- X/Y plot
- Textual Inversion
- Extras tab with:
  - GFPGAN, neural network that fixes faces
  - RealESRGAN, neural network upscaler
  - ESRGAN, neural network with a lot of third party models
- Resizing aspect ratio options
- Sampling method selection
- Interrupt processing at any time
- 4GB video card support
- Correct seeds for batches
- Prompt length validation
- Generation parameters added as text to PNG
- Tab to view an existing picture's generation parameters
- Settings page
- Running custom code from UI
- Mouseover hints for most UI elements
- Possible to change defaults/mix/max/step values for UI elements via text config
- Random artist button
- Tiling support: UI checkbox to create images that can be tiled like textures
- Progress bar and live image generation preview
- Negative prompt
- Styles
- Variations
- Seed resizing
- CLIP interrogator

## Stable Diffusion

*Stable Diffusion was made possible thanks to a collaboration with [Stability AI](https://stability.ai/) and [Runway](https://runwayml.com/) and builds upon previous work.*


Stable Diffusion is a latent text-to-image diffusion model.
Thanks to a generous compute donation from [Stability AI](https://stability.ai/) and support from [LAION](https://laion.ai/), they were able to train a Latent Diffusion Model on 512x512 images from a subset of the [LAION-5B](https://laion.ai/blog/laion-5b/) database. 
Similar to Google's [Imagen](https://arxiv.org/abs/2205.11487), this model uses a frozen CLIP ViT-L/14 text encoder to condition the model on text prompts.
With its 860M UNet and 123M text encoder, the model is relatively lightweight and runs on a GPU with at least 10GB VRAM.

## Comments 

- The codebase for the diffusion models builds heavily on [OpenAI's ADM codebase](https://github.com/openai/guided-diffusion)
and [https://github.com/lucidrains/denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch). 

- The implementation of the transformer encoder is from [x-transformers](https://github.com/lucidrains/x-transformers) by [lucidrains](https://github.com/lucidrains?tab=repositories). 


## BibTeX

```
@misc{rombach2021highresolution,
      title={High-Resolution Image Synthesis with Latent Diffusion Models}, 
      author={Robin Rombach and Andreas Blattmann and Dominik Lorenz and Patrick Esser and Björn Ommer},
      year={2021},
      eprint={2112.10752},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

```


