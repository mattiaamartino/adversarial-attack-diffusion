import torch
import torch.nn as nn
import clip
import PIL
from diffusers import StableDiffusionInstructPix2PixPipeline
from learnable_prompt import LearnablePrompt
from transformers import AutoImageProcessor, AutoModel

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
                 dinov2_processor=None,
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
            
        # Freeze pix2pix model weights - freeze each component separately
        for component_name in ['unet', 'text_encoder', 'vae', 'scheduler', 'feature_extractor', 'tokenizer', 'safety_checker', 'image_encoder']:
            if hasattr(self.pix2pix, component_name):
                component = getattr(self.pix2pix, component_name)
                if hasattr(component, 'parameters'):
                    for param in component.parameters():
                        param.requires_grad = False
            
        # Load or use provided DINOv2 model
        if dinov2_model is None:
            self.dinov2 = AutoModel.from_pretrained("facebook/dinov2-small")
            self.dinov2_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
            self.dinov2.to(self.device)
            self.dinov2.eval()
            
        else:
            self.dinov2 = dinov2_model
            self.dinov2_processor = dinov2_processor
            self.dinov2.eval()

         # Freeze DINOv2 model weights
        for param in self.dinov2.parameters():
            param.requires_grad = False

    
    def forward(self, image, num_inference_steps=50, guidance_scale=7.5):
        """
        Process an image through the attack pipeline.
        """
        # Get prompt embeddings
        positive_embeds = self.positive_prompt()
        negative_embeds = self.negative_prompt()
        
        # Generate modified image with pix2pix using learnable prompts
        output = self.pix2pix(
            prompt_embeds=positive_embeds,
            negative_prompt_embeds=negative_embeds,
            image=image,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            output_type="pt",  # Return tensor instead of PIL image
        )
        
        # Get the generated image tensor
        modified_image_tensor = output.images[0]  # This is now a tensor
        
        # Process original image through DINOv2
        # Convert PIL image to tensor if necessary
        if not isinstance(image, torch.Tensor):
            original_inputs = self.dinov2_processor(images=image, return_tensors="pt")
            original_pixel_values = original_inputs.pixel_values.to(self.device)
        else:
            original_pixel_values = image.unsqueeze(0) if image.dim() == 3 else image
            
        # Process original image
        original_outputs = self.dinov2(pixel_values=original_pixel_values)
        original_features = original_outputs.last_hidden_state[:, 0]  # Using CLS token
        
        # Process modified image through DINOv2
        # modified_image_tensor is already a tensor, just need to ensure correct format
        modified_pixel_values = modified_image_tensor.unsqueeze(0) if modified_image_tensor.dim() == 3 else modified_image_tensor
        modified_outputs = self.dinov2(pixel_values=modified_pixel_values)
        modified_features = modified_outputs.last_hidden_state[:, 0]  # Using CLS token
        
        # Convert modified image tensor to PIL for visualization if needed
        modified_image_pil = self.tensor_to_pil(modified_image_tensor)
            
        return modified_features, original_features, modified_image_pil
    
    def tensor_to_pil(self, tensor):
        """Helper function to convert tensor to PIL image for visualization"""
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        tensor = tensor.cpu().detach()
        tensor = (tensor + 1) / 2  # Convert from [-1, 1] to [0, 1]
        tensor = tensor.clamp(0, 1)
        tensor = tensor.permute(1, 2, 0).numpy()
        return PIL.Image.fromarray((tensor * 255).astype('uint8'))

    def get_trainable_parameters(self):
        """
        Returns the trainable parameters of the model (only the learnable prompts).
        """
        return list(self.positive_prompt.parameters()) + list(self.negative_prompt.parameters())

    def print_parameter_count(self):
        """
        Prints the number of trainable parameters vs total parameters in the network.
        """
        total_params = 0
        trainable_params = 0
        
        # Count parameters in all modules
        for name, module in self.named_modules():
            for param in module.parameters(recurse=False):
                total_params += param.numel()
                if param.requires_grad:
                    trainable_params += param.numel()
                    
        print(f"Trainable parameters: {trainable_params:,} / {total_params:,} total parameters")
        print(f"Percentage of trainable parameters: {100 * trainable_params / total_params:.2f}%")
