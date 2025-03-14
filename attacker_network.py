import torch
import torch.nn as nn
import clip
from diffusers import StableDiffusionInstructPix2PixPipeline
from learnable_prompt import LearnablePrompt
from prompt_tuning_attack.src.models.dinov2_model import Dinov2Model
from PIL import Image
from torchvision import transforms

class AttackerNetwork(nn.Module):
    """
    A network that combines learnable prompts with Instruct-pix2pix and DINOv2 models.
    
    This network:
    1. Initializes two LearnablePrompt objects (positive and negative)
    2. Injects them into a frozen Instruct-pix2pix model
    3. Processes the output through DINOv2
    4. Returns the features of the modified image, features of the original image, and the modified image
    """
    def __init__(self, 
                 clip_model=None,
                 pix2pix_model=None,
                 dinov2_model=None,
                 positive_template="Make the image: ", 
                 negative_template="bad quality, blurry, low resolution",
                 positive_ctx_len=10,
                 negative_ctx_len=5,
                 device=None):
        """
        Initialize the AttackerNetwork.
        
        Args:
            clip_model: Pre-loaded CLIP model (if None, will be loaded)
            pix2pix_model: Pre-loaded Instruct-pix2pix model (if None, will be loaded)
            dinov2_model: Pre-loaded DINOv2 model (if None, will be loaded)
            positive_template: Template string for positive prompt
            negative_template: Template string for negative prompt
            positive_ctx_len: Context length for positive prompt
            negative_ctx_len: Context length for negative prompt
            device: Device to run the models on (if None, will use CUDA if available)
        """
        super().__init__()
        
        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        # Load or use provided CLIP model
        if clip_model is None:
            self.clip_model, _ = clip.load("ViT-L/14", device=self.device)
        else:
            self.clip_model = clip_model
        
        # Initialize learnable prompts
        self.positive_prompt = LearnablePrompt(
            device=self.device,
            clip_model=self.clip_model,
            template=positive_template,
            ctx_len=positive_ctx_len
        )
        
        self.negative_prompt = LearnablePrompt(
            device=self.device,
            clip_model=self.clip_model,
            template=negative_template,
            ctx_len=negative_ctx_len
        )
        
        # Load or use provided pix2pix model
        if pix2pix_model is None:
            self.pix2pix = StableDiffusionInstructPix2PixPipeline.from_pretrained(
                "timbrooks/instruct-pix2pix",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                safety_checker=None,
            ).to(self.device)
        else:
            self.pix2pix = pix2pix_model
            
        # Freeze pix2pix model weights
        for param in self.pix2pix.parameters():
            param.requires_grad = False
            
        # Load or use provided DINOv2 model
        if dinov2_model is None:
            self.dinov2 = Dinov2Model()
            self.dinov2.load_model()
            self.dinov2.to(self.device)
            self.dinov2.eval()
            
            # Freeze DINOv2 model weights
            for param in self.dinov2.model.parameters():
                param.requires_grad = False
        else:
            self.dinov2 = dinov2_model
            
        # Image preprocessing for DINOv2
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225]),
        ])
    
    def forward(self, image, num_inference_steps=50, guidance_scale=7.5):
        """
        Process an image through the attack pipeline.
        
        Args:
            image: PIL Image to process
            num_inference_steps: Number of denoising steps for pix2pix
            guidance_scale: Guidance scale for pix2pix
            
        Returns:
            tuple: (modified_image_features, original_image_features, modified_image)
                - modified_image_features: DINOv2 features of the modified image
                - original_image_features: DINOv2 features of the original image
                - modified_image: The modified PIL Image
        """
        # Get prompt embeddings
        positive_embeds = self.positive_prompt()
        negative_embeds = self.negative_prompt()
        
        # Generate modified image with pix2pix using learnable prompts
        with torch.no_grad():
            output = self.pix2pix(
                prompt_embeds=positive_embeds,
                negative_prompt_embeds=negative_embeds,
                image=image,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
            )
        
        # Get the generated image
        modified_image = output.images[0]
        
        # Process original image through DINOv2
        original_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            original_features = self.dinov2.predict(original_tensor)
        
        # Process modified image through DINOv2
        modified_tensor = self.preprocess(modified_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            modified_features = self.dinov2.predict(modified_tensor)
            
        return modified_features, original_features, modified_image
    
    def get_trainable_parameters(self):
        """
        Returns the trainable parameters of the model (only the learnable prompts).
        """
        return list(self.positive_prompt.parameters()) + list(self.negative_prompt.parameters())
