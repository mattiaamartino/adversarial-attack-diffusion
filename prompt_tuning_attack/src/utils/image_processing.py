def load_image(image_path):
    from PIL import Image
    return Image.open(image_path)

def resize_image(image, size):
    return image.resize(size)

def normalize_image(image):
    import numpy as np
    image_array = np.array(image) / 255.0
    return image_array