# Hugging Face Vision Project

This project utilizes Hugging Face models to perform various vision tasks using the `facebook/dinov2-small` and `timbrooks/instruct-pix2pix` models. 

## Overview

The Hugging Face Vision Project is designed to explore and demonstrate the capabilities of two powerful models in the field of computer vision. The `dinov2` model is used for feature extraction and representation learning, while the `pix2pix` model is employed for image generation tasks.

## Installation

To set up the project, clone the repository and install the required dependencies:

```bash
git clone <repository-url>
cd huggingface-vision-project
pip install -r requirements.txt
```

## Project Structure

- `src/`: Contains the source code for the project.
  - `models/`: Includes model definitions and implementations.
    - `dinov2_model.py`: Class for loading and using the `facebook/dinov2-small` model.
    - `pix2pix_model.py`: Class for loading and using the `timbrooks/instruct-pix2pix` model.
  - `utils/`: Contains utility functions for image processing and visualization.
  - `config.py`: Configuration settings for model paths and hyperparameters.
  - `main.py`: Entry point for running the application.

- `notebooks/`: Jupyter notebooks for exploring and demonstrating the models.
  - `dinov2_exploration.ipynb`: Notebook for exploring the `dinov2` model.
  - `pix2pix_examples.ipynb`: Notebook for demonstrating the `pix2pix` model.

- `data/`: Directory for storing raw and processed data.
  - `raw/`: Contains raw data files.
  - `processed/`: Contains processed data files.

- `requirements.txt`: Lists the dependencies required for the project.

- `setup.py`: Setup script for the project.

## Usage

After installing the dependencies, you can run the main application or explore the Jupyter notebooks to see the models in action.

To run the main application, execute:

```bash
python src/main.py
```

For detailed examples and visualizations, open the Jupyter notebooks in the `notebooks/` directory.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.