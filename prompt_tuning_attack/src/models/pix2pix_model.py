import torch

class Pix2PixModel:
    def __init__(self):
        self.model = None

    def load_model(self):
        from transformers import AutoModelForImageToImageTranslation
        self.model = AutoModelForImageToImageTranslation.from_pretrained("timbrooks/instruct-pix2pix")

    def preprocess_input(self, input_image):
        from PIL import Image
        import torchvision.transforms as transforms

        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])
        return transform(input_image).unsqueeze(0)

    def generate_image(self, input_image):
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        input_tensor = self.preprocess_input(input_image)
        with torch.no_grad():
            generated_image = self.model(input_tensor).squeeze(0)
        
        return generated_image