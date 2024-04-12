import json
import pickle
import random
import os
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense,Dropout
import numpy as np
from sklearn.preprocessing import StandardScaler


import nltk
from nltk.stem import WordNetLemmatizer

lemmatizer=WordNetLemmatizer()
intents=json.loads(open('D:\\chatbot\\src\intents.json').read())

words=[]
documents=[]
classes=[]
ignoreletters=['?', '!', '.', ',']


for intent in intents['intents']:
    for patterns in intent['patterns']:
        wordlist=nltk.word_tokenize(patterns)
        words.extend(wordlist)
        documents.append((wordlist, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words=[lemmatizer.lemmatize(word) for word in words if word not in ignoreletters]
words=sorted(set(words))

classes=sorted(set(classes))

pickle.dump(words,open('words.pkl', 'wb'))
pickle.dump(classes,open('classes.pkl','wb'))

#NOW CREATE a training set with classes and words

training=[]
outputempty=[0] * len(classes)

for document in documents:
    bag=[]
    wordpattern=document[0]
    wordpattern=[lemmatizer.lemmatize(word.lower()) for word in wordpattern]
    for word in words:
        bag.append(1) if word in wordpattern else bag.append(0)
    outputrow=list(outputempty)
    outputrow[classes.index(document[1])]=1
    training.append(bag+outputrow)
    
random.shuffle(training)
training=np.array(training)

trainx = training[:, :len(words)]
trainy = training[:, len(words):]


split_index = int(0.8 * len(trainx))
testx, valx = trainx[:split_index], trainx[split_index:]
testy, valy = trainy[:split_index], trainy[split_index:]
    
#model for the chatbot trainging
# Define the model
scaler = StandardScaler()
trainx_normalized = scaler.fit_transform(trainx)
# Define the model
model = tf.keras.Sequential([
    Dense(64, input_shape=(len(trainx[0]),)),
    tf.keras.layers.ReLU(),
    Dropout(0.2),
    Dense(36),
    tf.keras.layers.ReLU(),
    Dropout(0.2),
    Dense(len(trainy[0]), activation='softmax')
])

# Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
history = model.fit(trainx_normalized, np.array(trainy), epochs=100, batch_size=32, validation_split=0.1, verbose=1)

# Evaluate the model
loss, accuracy = model.evaluate(trainx_normalized, np.array(trainy), verbose=0)
print(f'Train loss: {loss}')
print(f'Train accuracy: {accuracy}')

# Save the model
model.save('chatbot_model.h5')
print('Model saved')


            

            
            

