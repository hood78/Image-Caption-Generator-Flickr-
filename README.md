# flickr-image-caption-generator/flickr-image-caption-generator/README.md

# Flickr Image Caption Generator

This project implements an image caption generator using the Flickr dataset. The model is built using Keras and TensorFlow frameworks, leveraging a pre-trained CNN for feature extraction and an encoder-decoder architecture with attention mechanisms for generating captions.

## Project Structure

flickr-image-caption-generator
├── data
│   ├── images          # Contains images from the Flickr dataset
│   └── captions        # Contains caption files associated with the images
├── src
│   ├── data_preprocessing.py  # Functions for preprocessing the dataset
│   ├── feature_extraction.py   # Extracts features from images using InceptionV3
│   ├── model.py                # Defines the encoder-decoder model with attention
│   ├── train.py                # Trains the model on image-caption pairs
│   ├── evaluate.py             # Evaluates the model performance
│   └── utils.py                # Utility functions for data handling and model saving
├── notebooks
│   └── exploration.ipynb       # Jupyter notebook for exploratory data analysis
├── requirements.txt            # Lists project dependencies
├── README.md                   # Project documentation
└── .gitignore                  # Specifies files to ignore in Git


## Setup Instructions

1. Clone the repository:

   git clone <repository-url>
   cd flickr-image-caption-generator
  

2. Install the required dependencies:
   
   pip install -r requirements.txt
   

3. Download the Flickr dataset and place the images in the `data/images` directory and the captions in the `data/captions` directory.

## Usage Guidelines

- Preprocess the data by running `src/data_preprocessing.py`.
- Extract features from the images using `src/feature_extraction.py`.
- Train the model with `src/train.py`.
- Evaluate the model's performance using `src/evaluate.py`.
- Use the Jupyter notebook in `notebooks/exploration.ipynb` for data visualization and analysis.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.
