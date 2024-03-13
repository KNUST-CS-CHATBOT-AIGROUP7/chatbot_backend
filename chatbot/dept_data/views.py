from django.shortcuts import render
from django.http import HttpResponse
from rest_framework.views import APIView
import json
import pickle
import numpy as np
import nltk
import random
from django.shortcuts import redirect

from nltk.stem import WordNetLemmatizer
from keras.models import load_model

file_path = "C:\\Users\\Owusu-Turkson\\Documents\\GitHub\\chatbot_backend\\chatbot\\dept_data\\intents.json" 

lemmatizer = WordNetLemmatizer()

try:
    with open(file_path) as file:
        intents = json.load(file)
except FileNotFoundError:
    exit("File not found")

# URL of the pickle file
url_words = 'C:\\Users\\Owusu-Turkson\\Documents\\GitHub\\chatbot_backend\\chatbot\\dept_data\\words.pkl'
url_classes = 'C:\\Users\\Owusu-Turkson\\Documents\\GitHub\\chatbot_backend\\chatbot\\dept_data\\classes.pkl'


words = pickle.load(open(url_words, 'rb'))
classes = pickle.load(open(url_classes, 'rb'))
model = load_model('C:\\Users\\Owusu-Turkson\\Documents\\GitHub\\chatbot_backend\\chatbot\\dept_data\\chatbot_model.keras')

# Create your views here.
class Files:
    def clean_up_sentence(self, sentence):
        sentence_words = nltk.word_tokenize(sentence)
        sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
        return sentence_words

    def bag_of_words (self, sentence):
        sentence_words = self.clean_up_sentence(sentence)
        bag = [0] * len(words)
        for w in sentence_words:
            for i, word in enumerate(words):
                if word == w:
                    bag[i] = 1
        return np.array(bag)

    def predict_class (self, sentence):
        bow = self.bag_of_words (sentence)
        res = model.predict(np.array([bow]))[0]
        ERROR_THRESHOLD = 0.25
        results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in results:
            return_list.append({'intent': classes [r[0]], 'probability': str(r[1])})
        return return_list

    def get_response(self, intents_list, intents_json):
        if intents_list:
            tag = intents_list[0]['intent']
            list_of_intents = intents_json['intents']
            for i in list_of_intents:
                if i['tag'] == tag:
                    result = random.choice (i['responses'])
                    break
            return result
        else:
            return "I'm sorry, I do not understand that"
class Chatbot(APIView):
       def __init__(self):
           super().__init__()
           self.files = Files() # create an instance of the Files class to use its methods

       def get(self, request):
           
           return redirect("")

       def post(self, request):
           data = request.data["user"]
           while data.replace(" ","") == "":
               continue
           print(data)
           ints = self.files.predict_class(data)
           res = self.files.get_response (ints, intents)
           return HttpResponse(res)
        #    return HttResponse(json.dumps(data))