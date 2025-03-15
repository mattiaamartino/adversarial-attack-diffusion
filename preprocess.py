import os

import numpy as np
from tqdm import tqdm
import umap.umap_ as umap
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.metrics.pairwise import cosine_distances

import torch
from torchvision import transforms
from torch.utils.data import Subset

device = "cuda" if torch.cuda.is_available() else "cpu"

seed = 0
np.random.seed(seed)
torch.manual_seed(seed)

pil_to_tensor = transforms.ToTensor()
tensor_to_pil = transforms.ToPILImage()

def compute_centroids(model, dataloader, use_precomputed=True, precomputed_dir='variables'):

    os.makedirs(precomputed_dir, exist_ok=True)

    centroids_path = os.path.join(precomputed_dir, 'centroids.npy')
    features_path = os.path.join(precomputed_dir, 'features.npy')
    labels_path = os.path.join(precomputed_dir, 'labels.npy')

    if not use_precomputed:
        features = []
        labels = []

        for images, label in tqdm(dataloader):
            images = images.to(device)
            with torch.no_grad():
                feature = model(images).last_hidden_state[:, 0]
            feature = feature.cpu().numpy()
            features.append(feature)
            labels.extend(label.cpu().numpy())
        
        features = np.concatenate(features, axis=0)
        labels = np.array(labels)

        unique_classes = np.unique(labels)
        n_classes = len(unique_classes)

        centroids = np.zeros((n_classes, features.shape[1]))
        for i, c in enumerate(unique_classes):
            class_indices = np.where(labels == c)[0]
            centroids[i] = np.mean(features[class_indices], axis=0)

        np.save(centroids_path, centroids)
        np.save(features_path, features)
        np.save(labels_path, labels)

    else:
        centroids = np.load(centroids_path)
        features = np.load(features_path)
        labels = np.load(labels_path)

    return centroids, features, labels

def select_starting_target_classes(labels, dataset):
    
    unique_classes = np.unique(labels)

    global class_id_to_name
    class_id_to_name = {v: k for k, v in dataset.class_to_idx.items()}

    selected_classes = np.random.choice(unique_classes, size=2, replace=False)
    starting_class = selected_classes[0]
    starting_class_name = class_id_to_name[starting_class].split('-')[-1]
    target_class = selected_classes[1]
    target_class_name = class_id_to_name[target_class].split('-')[-1]

    print(f"Starting class: {starting_class_name}")
    print(f"Target class: {target_class_name}")

    return starting_class, target_class


def plot_umap(features, labels, subset_size=100, plot_dir='plots'):

    os.makedirs(plot_dir, exist_ok=True)

    starting_class, target_class, name_dict = select_starting_target_classes(labels)

    classes = {starting_class: name_dict[starting_class].split('-')[-1] + " (starting)",
                target_class: name_dict[target_class].split('-')[-1] + " (target)"}

    starting_features = []
    target_features = []
    for c in [starting_class, target_class]:
        class_indices = np.where(labels == c)[0]
        selected_indices = np.random.choice(class_indices, size=subset_size, replace=False)
        if c == starting_class:
            starting_features.append(features[selected_indices])
        else:
            target_features.append(features[selected_indices])

    starting_features = np.concatenate(starting_features, axis=0)
    target_features = np.concatenate(target_features, axis=0)
    plotting_features = np.concatenate([starting_features, target_features], axis=0)

    starting_labels = np.full(starting_features.shape[0], starting_class)
    target_labels = np.full(target_features.shape[0], target_class)
    plotting_labels = np.concatenate([starting_labels, target_labels], axis=0)

    umap_reducer = umap.UMAP(n_components=2, random_state=seed)
    embedding = umap_reducer.fit_transform(plotting_features)

    colors = cm.rainbow(np.linspace(0, 1, len([starting_class, target_class])))

    plt.figure(figsize=(10, 8))
    for i, cls in enumerate([starting_class, target_class]):
        cls_mask = (plotting_labels == cls)
        plt.scatter(
            embedding[cls_mask, 0],
            embedding[cls_mask, 1],
            s=50,
            color=colors[i],
            label=classes[cls]
        )

    plt.title("UMAP Projection of Image Features for Starting Class and Target Class")
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.legend()
    try:
        __file__
        plt.savefig(os.path.join(plot_dir, 'umap_projection.png'))
        plt.close()
    except NameError:
        plt.show()

def compute_initial_metrics(centroids, starting_class, target_class):

    starting_centroid = centroids[starting_class]
    target_centroid = centroids[target_class]

    # Compute the distance between the centroids
    distance = cosine_distances(starting_centroid.reshape(1, -1), target_centroid.reshape(1, -1))[0][0]
    print(f"Distance between centroids:         {distance:.2f}")

    distances = cosine_distances(centroids)
    average_distance = distances.mean()
    print(f"Average distance between centroids: {average_distance:.2f}")

    min_distance = distances[~np.eye(distances.shape[0], dtype=bool)].min()
    print(f"Minimum distance between centroids: {min_distance:.2f}")

    max_distance = distances[~np.eye(distances.shape[0], dtype=bool)].max()
    print(f"Maximum distance between centroids: {max_distance:.2f}")


def create_subset(dataset, starting_class, use_precomputed=True, precomputed_dir='variables'):
    os.makedirs(precomputed_dir, exist_ok=True)

    subset_path = os.path.join(precomputed_dir, 'target_indices.npy')

    if not use_precomputed:
        subset_indices = [i for i, (_, label) in enumerate(dataset) 
                          if label == starting_class]
        
    else:
        subset_indices = torch.load(subset_path)
    
    subset = Subset(dataset, subset_indices)

    return subset

def preprocess(
        encoder_model,
        dataloader,
        dataset,
        use_precomputed_centroids=True,
        use_precomputed_subset=True,
        plot_umap_projection=True,
        variable_dir='variables',
        plot_dir='plots',
        subset_size=100,
        ):
    
    # Compute centroids
    centroids, features, labels = compute_centroids(
        encoder_model,
        dataloader,
        use_precomputed=use_precomputed_centroids,
        precomputed_dir=variable_dir
    )

    # Select starting and target classes
    starting_class, target_class = select_starting_target_classes(labels, dataset)

    if plot_umap_projection:
        plot_umap(features, labels, subset_size=subset_size, plot_dir=plot_dir)

    # Compute initial metrics
    compute_initial_metrics(centroids, starting_class, target_class)

    # Create subset of dataset
    subset = create_subset(
        dataset,
        starting_class,
        target_class,
        use_precomputed=use_precomputed_subset,
        precomputed_dir=variable_dir
    )
    print()
    print(f"Subset size: {len(subset)}")
    print()

    return subset, centroids, starting_class, target_class
