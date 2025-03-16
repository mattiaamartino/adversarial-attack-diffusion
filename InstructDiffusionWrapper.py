import sys

import torch.utils.checkpoint
sys.path.append("/users/eleves-b/2024/mattia.martino/adversarial-attack-diffusion/InstructDiffusion/stable_diffusion")

import torch
import torch.nn as nn
from torch import autocast
import torch.nn.functional as F

from omegaconf import OmegaConf
import einops
import math

from InstructDiffusion.stable_diffusion.ldm.util import instantiate_from_config
import k_diffusion as K



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class InstructDiffusion(nn.Module):
    """
    A pipeline wrapper for InstructDiffusion that mimics the interface of 
    StableDiffusionInstructPix2PixPipeline for compatibility.
    """
    def __init__(self, config_path, ckpt_path, device="cuda"):
        super().__init__()
        self.device = device
        
        # Load config
        config = OmegaConf.load(config_path)
        # Instantiate model skeleton
        self.model = load_model_from_config(config, ckpt_path)
        self.model = self.model.eval().to(device)

        for p in self.model.parameters():
           p.requires_grad = False
        
        self.model_wrap = K.external.CompVisDenoiser(self.model)
        self.model_wrap_cfg = CFGDenoiser(self.model_wrap)

    
    def forward(
        self,
        image,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        num_inference_steps=50,
        cfg_text=7.5,
        cfg_image=1.5,
    ):
    
        with autocast("cuda"):
            def encode(x):
                return self.model.encode_first_stage(2*x+1)
            image = self.preprocess_image(image)
            image_latent = encode(image)#torch.utils.checkpoint.checkpoint(encode, image)
            if hasattr(image_latent, "mode"):
                image_latent = image_latent.mode()

            cond = {
                "c_concat": [image_latent],
            }
            if prompt_embeds is not None:
                cond["c_crossattn"] = [prompt_embeds]
            else:
                cond["c_crossattn"] = [self.model.get_learned_conditioning(["Make it different"])]

            uncond = {
                "c_concat": [image_latent],
            }
            if negative_prompt_embeds is not None:
                uncond["c_crossattn"] = [negative_prompt_embeds]
            else:
                uncond["c_crossattn"] = [self.model.get_learned_conditioning([""])]

            sigmas = self.model_wrap.get_sigmas(num_inference_steps)
            noise = torch.randn_like(image_latent, requires_grad=True) * sigmas[0]
            z = noise.clone()
            print(z.requires_grad)
            for i in range(len(sigmas)-1):
                sigma = sigmas[i]
                if sigma.dim() == 0:
                    sigma = sigma.unsqueeze(0)
                def denoise(z):
                    return self.model_wrap_cfg(
                        z, sigma, 
                        cond, uncond, 
                        cfg_text, cfg_image
                    )
                
                z = denoise(z)#torch.utils.checkpoint.checkpoint(denoise, z)
                # Euler sampling
                z = z + torch.sqrt(sigmas[i]**2 - sigmas[i+1]**2) * torch.randn_like(z, requires_grad=True)
            def decode(z):
                return self.model.decode_first_stage(z)
            print(z.requires_grad)
            output = decode(z)#torch.utils.checkpoint.checkpoint(decode, z)
            print(output.requires_grad)
            output = torch.clamp((output + 1) / 2, 0.0, 1.0)
            print(output.requires_grad)
            return output
    

    def preprocess_image(self, image):
        _, _, w, h = image.shape
        factor = 224 / max(w, h)
        factor = math.ceil(min(w, h) * factor / 64) * 64 / min(w, h)
        w_resize = int((w * factor) // 64) * 64
        h_resize = int((h * factor) // 64) * 64

        image = F.interpolate(image, size=(h_resize, w_resize), mode="bilinear", align_corners=False)
        return image

    
def load_model_from_config(config, ckpt, vae_ckpt=None, verbose=False):
    model = instantiate_from_config(config.model)

    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if 'state_dict' in pl_sd:
        pl_sd = pl_sd['state_dict']
    m, u = model.load_state_dict(pl_sd, strict=False)

    print(m, u)
    return model
    
    
class CFGDenoiser(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, z, sigma, cond, uncond, text_cfg_scale, image_cfg_scale):
        cfg_z = einops.repeat(z, "b ... -> (repeat b) ...", repeat=3)
        cfg_sigma = einops.repeat(sigma, "b ... -> (repeat b) ...", repeat=3)
        cfg_cond = {
            "c_crossattn": [torch.cat([cond["c_crossattn"][0], uncond["c_crossattn"][0], cond["c_crossattn"][0]])],
            "c_concat": [torch.cat([cond["c_concat"][0], cond["c_concat"][0], uncond["c_concat"][0]])],
        }
        out_cond, out_img_cond, out_txt_cond \
            = self.inner_model(cfg_z, cfg_sigma, cond=cfg_cond).chunk(3)
        return 0.5 * (out_img_cond + out_txt_cond) + \
            text_cfg_scale * (out_cond - out_img_cond) + \
                image_cfg_scale * (out_cond - out_txt_cond)