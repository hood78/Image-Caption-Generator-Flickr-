import numpy as np
import pandas as pd
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
from nltk.translate.bleu_score import corpus_bleu
from src.utils import load_data, preprocess_image

def evaluate_model(model, image_path, tokenizer, max_length):
    # Load and preprocess the image
    image = preprocess_image(image_path)
    image = np.expand_dims(image, axis=0)

    # Generate a caption
    caption = generate_caption(model, image, tokenizer, max_length)
    return caption

def generate_caption(model, image, tokenizer, max_length):
    # Start with the start token
    in_text = '<start>'
    for _ in range(max_length):
        # Encode the text
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)

        # Predict the next word
        yhat = model.predict([image, sequence], verbose=0)
        yhat = np.argmax(yhat)

        # Map the predicted word index to word
        word = tokenizer.index_word[yhat]
        if word == '<end>':
            break
        in_text += ' ' + word
    return in_text

def evaluate_on_dataset(model, tokenizer, max_length, dataset_path):
    # Load the dataset
    captions = load_data(dataset_path)
    actual, predicted = [], []

    for image_id, caption_list in captions.items():
        image_path = f'data/images/{image_id}'
        predicted_caption = evaluate_model(model, image_path, tokenizer, max_length)
        actual.append([caption.split() for caption in caption_list])
        predicted.append(predicted_caption.split())

    # Calculate BLEU score
    bleu_score = corpus_bleu(actual, predicted)
    print(f'BLEU score: {bleu_score}')

if __name__ == '__main__':
    # Load the trained model
    model = load_model('model.h5')
    tokenizer = ...  # Load your tokenizer here
    max_length = ...  # Define your max_length here
    dataset_path = 'data/captions/captions.csv'  # Path to your captions dataset

    evaluate_on_dataset(model, tokenizer, max_length, dataset_path)