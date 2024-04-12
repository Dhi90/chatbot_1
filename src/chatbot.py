import random
import json
import pickle
import numpy as np
import nltk

from nltk.stem import WordNetLemmatizer
from keras.models import load_model

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('D:\chatbot\src\intents.json').read())

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')

#cleaning the message
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

#bagging thw words from message
def bag_of_words (sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

#predict the response of the message
def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.5
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    if not results:
        return [{'intent': 'unknown', 'probability': '1.0'}]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        if r[0] < len(classes):
            return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

#retrive the responce from the model
def get_response(intents_list, intents_json):
    if intents_list:
        tag = intents_list[0]['intent']
        list_of_intents = intents_json['intents']
        for i in list_of_intents:
            if i['tag'] == tag:
                return random.choice(i['responses'])
        # If no matching tag is found, return a default response
        return "I'm sorry, I didn't understand that."
    else:
        # If no intent is predicted above the threshold, return a default response
        return "I'm sorry, I didn't understand that."


print("GO! Bot is running!")

while True:
    message = input("")
    ints = predict_class (message)
    res = get_response (ints, intents)
    print (res)
    