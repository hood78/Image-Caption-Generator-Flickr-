import os
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def load_images(image_dir, target_size=(299, 299)):
    images = []
    for filename in os.listdir(image_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img_path = os.path.join(image_dir, filename)
            img = Image.open(img_path)
            img = img.resize(target_size)
            img_array = np.array(img) / 255.0  # Normalize pixel values
            images.append(img_array)
    return np.array(images)

def load_captions(captions_file):
    with open(captions_file, 'r') as file:
        captions = file.readlines()
    captions = [caption.strip() for caption in captions]
    return captions

def tokenize_captions(captions, max_length):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(captions)
    sequences = tokenizer.texts_to_sequences(captions)
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')
    return padded_sequences, tokenizer

def preprocess_data(image_dir, captions_file, max_length):
    images = load_images(image_dir)
    captions = load_captions(captions_file)
    tokenized_captions, tokenizer = tokenize_captions(captions, max_length)
    return images, tokenized_captions, tokenizer

# Example usage:
# images, captions, tokenizer = preprocess_data('data/images', 'data/captions/captions.txt', max_length=20)