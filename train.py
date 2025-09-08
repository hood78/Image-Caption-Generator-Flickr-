import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from model import create_model
from data_preprocessing import preprocess_data

def load_data(image_dir, caption_file):
    captions = pd.read_csv(caption_file)
    image_names = captions['image'].values
    captions_list = captions['caption'].values
    return image_names, captions_list

def train_model(image_dir, caption_file, epochs=20, batch_size=32):
    image_names, captions_list = load_data(image_dir, caption_file)
    features, tokenizer = preprocess_data(image_names, captions_list)

    model = create_model(vocab_size=len(tokenizer.word_index) + 1)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

    checkpoint = ModelCheckpoint('model.h5', save_best_only=True, monitor='val_loss', mode='min')

    model.fit(features, captions_list, epochs=epochs, batch_size=batch_size, callbacks=[checkpoint])

if __name__ == "__main__":
    image_directory = os.path.join('data', 'images')
    caption_file_path = os.path.join('data', 'captions', 'captions.csv')
    train_model(image_directory, caption_file_path)