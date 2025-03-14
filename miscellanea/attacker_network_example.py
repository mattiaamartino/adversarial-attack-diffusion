import torch
import clip
from diffusers import StableDiffusionInstructPix2PixPipeline
from prompt_tuning_attack.src.models.dinov2_model import Dinov2Model
from prompt_tuning_attack.src.utils.visualization import open_image, display_image
from attacker_network import AttackerNetwork
from PIL import Image

def main():
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load models (these will be shared across multiple AttackerNetwork instances)
    print("Loading CLIP model...")
    clip_model, _ = clip.load("ViT-L/14", device=device)
    
    print("Loading Instruct-pix2pix model...")
    pix2pix = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        "timbrooks/instruct-pix2pix",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        safety_checker=None,
    ).to(device)
    
    print("Loading DINOv2 model...")
    dinov2 = Dinov2Model()
    dinov2.load_model()
    dinov2.to(device)
    dinov2.eval()
    
    # Create AttackerNetwork instance
    print("Creating AttackerNetwork...")
    attacker = AttackerNetwork(
        clip_model=clip_model,
        pix2pix_model=pix2pix,
        dinov2_model=dinov2,
        positive_template="Make the image: ",
        negative_template="bad quality, blurry, low resolution",
        positive_ctx_len=10,
        negative_ctx_len=5,
        device=device
    )
    
    # Load an example image
    image_path = "zebra.jpg"  # Update with your image path
    print(f"Loading image from {image_path}...")
    image = open_image(image_path)
    
    # Process the image through the attacker network
    print("Processing image through AttackerNetwork...")
    modified_features, original_features, modified_image = attacker(
        image, 
        num_inference_steps=20,  # Reduced for faster execution
        guidance_scale=7.5
    )
    
    # Display results
    print("Modified image features shape:", modified_features[0].shape)
    print("Original image features shape:", original_features[0].shape)
    
    # Save the modified image
    output_path = "modified_zebra.jpg"
    modified_image.save(output_path)
    print(f"Modified image saved to {output_path}")
    
    # Get trainable parameters
    trainable_params = attacker.get_trainable_parameters()
    print(f"Number of trainable parameters: {sum(p.numel() for p in trainable_params)}")
    
    # Example of creating multiple attackers that share the same models
    print("\nCreating multiple AttackerNetwork instances with shared models...")
    attackers = []
    for i in range(3):
        attacker = AttackerNetwork(
            clip_model=clip_model,
            pix2pix_model=pix2pix,
            dinov2_model=dinov2,
            positive_template=f"Template {i}: Make the image",
            positive_ctx_len=5 + i,
            device=device
        )
        attackers.append(attacker)
    
    print(f"Created {len(attackers)} AttackerNetwork instances that share the same models")

if __name__ == "__main__":
    main()
