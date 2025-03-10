from torchvision import transforms
import torch

class Dinov2Model:
    def __init__(self):
        self.model = None

    def load_model(self):
        from transformers import AutoModel
        self.model = AutoModel.from_pretrained("facebook/dinov2-small")

    def preprocess_input(self, image):

        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225]),
        ])
        return preprocess(image).unsqueeze(0)

    def predict(self, image):
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        with torch.no_grad():
            outputs = self.model(image)
        return outputs
    
    def eval(self):
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        self.model.eval()

    def to(self, device):
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        self.model.to(device)