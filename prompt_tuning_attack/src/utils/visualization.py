def display_image(image, title=None):
    import matplotlib.pyplot as plt
    
    plt.imshow(image)
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()

def plot_results(original, generated, title='Results'):
    import matplotlib.pyplot as plt
    
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