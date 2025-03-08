from models.dinov2_model import Dinov2Model
from models.pix2pix_model import Pix2PixModel
from utils.image_processing import load_image, resize_image, normalize_image
from utils.visualization import display_image, plot_results
import config

def main():
    # Initialize models
    dinov2_model = Dinov2Model()
    pix2pix_model = Pix2PixModel()

    # Load models
    dinov2_model.load_model()
    pix2pix_model.load_model()

    # Example input for Dinov2
    input_image = load_image("path/to/input/image.jpg")
    processed_image = dinov2_model.preprocess_input(input_image)
    dinov2_prediction = dinov2_model.predict(processed_image)

    # Visualize Dinov2 results
    display_image(dinov2_prediction)
    
    # Example input for Pix2Pix
    pix2pix_input = resize_image(input_image, config.IMAGE_DIMENSIONS)
    pix2pix_processed = pix2pix_model.preprocess_input(pix2pix_input)
    generated_image = pix2pix_model.generate_image(pix2pix_processed)

    # Visualize Pix2Pix results
    plot_results(input_image, generated_image)

if __name__ == "__main__":
    main()