import matplotlib.pyplot as plt
from PIL import Image

def open_image(image_path):
    image = Image.open(image_path)
    image = image.convert("RGB")  # Ensure image is in RGB

    return image

def display_image(image, title=None):
    plt.imshow(image)
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()


def plot_results(original, generated, title='Results'):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(original)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(generated)
    plt.title('Generated Image')
    plt.axis('off')
    
    plt.suptitle(title)
    plt.show()