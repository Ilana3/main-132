import tensorflow.keras
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
sentence = ["Eu estou feliz em encontrar os meus amigos. Estamos planejandoir a uma festa.",
            "Hoje o dia foi ruim na escola. Eu me machuquei jogando futebol"]

tokenizer = Tokenizer(num_words=10000, oov_token='OOV')
tokenizer.fit_on_texts(sentence)

word_index = tokenizer.word_index
sequence = tokenizer.text_to_sequences(sentence)
print(sequence[0:2])

padded = pad_sequences(sequence, maxlen = 100, padding='post',
                       truncating='post')
print(padded[0:2])

model=tensorflow.keras.models.load_model('Text_Emoticon.h5')

result=model.predict(padded)
print(result)

predict_class = np.argmax(result, axis=1)
print(predict_class)
{"raiva": 0, "medo": 1, "alegria": 2, "amor": 3, "tristeza": 4, "surpresa": 5}