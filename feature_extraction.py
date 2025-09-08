from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

def extract_features(directory):
    model = InceptionV3(weights='imagenet', include_top=False)
    model = model.predict(np.zeros((1, 299, 299, 3)))  # Dummy prediction to load the model

    features = {}
    for filename in os.listdir(directory):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img_path = os.path.join(directory, filename)
            image = load_img(img_path, target_size=(299, 299))
            image = img_to_array(image)
            image = np.expand_dims(image, axis=0)
            image = image / 255.0  # Normalize the image
            feature = model.predict(image)
            features[filename] = feature.flatten()  # Flatten the feature vector

    return features

if __name__ == "__main__":
    directory = '../data/images'  # Adjust the path as necessary
    features = extract_features(directory)
    print("Extracted features for {} images.".format(len(features)))