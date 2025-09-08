from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Add, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

class ImageCaptioningModel:
    def __init__(self, vocab_size, max_length):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()

    def build_encoder(self):
        base_model = InceptionV3(weights='imagenet', include_top=False)
        base_model.trainable = False
        inputs = Input(shape=(299, 299, 3))
        features = base_model(inputs)
        features = Dense(256, activation='relu')(features)
        model = Model(inputs, features)
        return model

    def build_decoder(self):
        inputs1 = Input(shape=(256,))
        inputs2 = Input(shape=(self.max_length,))
        embedding = Embedding(self.vocab_size, 256)(inputs2)
        lstm = LSTM(256)(embedding)
        decoder_output = Add()([inputs1, lstm])
        decoder_output = Dense(256, activation='relu')(decoder_output)
        outputs = Dense(self.vocab_size, activation='softmax')(decoder_output)
        model = Model(inputs=[inputs1, inputs2], outputs=outputs)
        return model

    def compile_model(self):
        self.decoder.compile(loss='categorical_crossentropy', optimizer='adam')

    def predict(self, image_feature, caption_sequence):
        caption_sequence = np.array(caption_sequence).reshape(1, -1)
        return self.decoder.predict([image_feature, caption_sequence])