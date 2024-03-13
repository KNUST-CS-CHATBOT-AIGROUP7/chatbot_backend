from django.shortcuts import render
from django.http import HttpResponse
from rest_framework.views import APIView
import json
import pickle
import numpy as np
import nltk
import random

from nltk.stem import WordNetLemmatizer
from keras.models import load_model

file_path = "chatbot\dept_data\intents.json" 

lemmatizer = WordNetLemmatizer()

try:
    with open(file_path) as file:
        intents = json.load(file)
except FileNotFoundError:
    print("File not found")

words = pickle.load(open('https://github.com/KNUST-CS-CHATBOT-AIGROUP7/chatbot_backend/blob/dev/chatbot/dept_data/words.pkl', 'rb'))
classes = pickle.load(open('https://github.com/KNUST-CS-CHATBOT-AIGROUP7/chatbot_backend/blob/dev/chatbot/dept_data/classes.pkl', 'rb'))
model = load_model('https://github.com/KNUST-CS-CHATBOT-AIGROUP7/chatbot_backend/blob/dev/chatbot/dept_data/chatbot_model.keras')


# Create your views here.
class Files():
    def clean_up_sentence(sentence):
        sentence_words = nltk.word_tokenize(sentence)
        sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
        return sentence_words

    def bag_of_words (sentence):
        sentence_words = Files().clean_up_sentence(sentence)
        bag = [0] * len(words)
        for w in sentence_words:
            for i, word in enumerate(words):
                if word == w:
                    bag[i] = 1
        return np.array(bag)

    def predict_class (sentence):
        bow = Files().bag_of_words (sentence)
        res = model.predict(np.array([bow]))[0]
        ERROR_THRESHOLD = 0.25
        results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in results:
            return_list.append({'intent': classes [r[0]], 'probability': str(r[1])})
        return return_list

    def get_response(intents_list, intents_json):
        tag = intents_list[0]['intent']
        list_of_intents = intents_json['intents']
        for i in list_of_intents:
            if i['tag'] == tag:
                result = random.choice (i['responses'])
                break
        return result

class Chatbot(APIView):
       def post(self, request, *args, **kwargs):
           data = request.data
        #    message = input("")
        #    message= data.get( "message", "")
           ints = Files().predict_class (data)
           res = Files().get_response (ints, intents)
           return HttpResponse(json.dumps({"response":res}))
        #    return HttpResponse(json.dumps(data))
       



# class Files():
#     def main(self):
#         return "Hello, world. You're at the dept_data index."
#     def get_response(intents_list, intents_json):
#          pass
#     def predict_class(self, sentence):
#         pass
# class Chatbot(APIView):
#        def post(self, request, *args, **kwargs):
#            data = request.data
#            ints= Files().predict_class(data)
#            res = get_response(ints,intents)
#            return HttpResponse(json.dumps(data))
