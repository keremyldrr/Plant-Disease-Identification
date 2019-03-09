import os
import numpy as np
import pprint
import cv2 as cv
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from flask import Flask
import json
from flask_restful import reqparse, abort, Api, Resource
app = Flask(__name__)
api = Api(app)

class Plant(Resource):
    def get(self):
        return json.dumps("Please enter image path")

class Plant_Disease(Resource):
    def get(self,image_path=""):
        #print(image_path)
        if image_path!="favicon.ico":
            json_file = open('model.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            model = tf.keras.models.model_from_json(loaded_model_json)
            model.load_weights("weights.h5")
            img = plt.imread(image_path[:])
            #print(img.shape)
            img = np.resize(img,(256,256,3))
            img = np.expand_dims(img,axis=0)        
            pred = np.argmax(model.predict(img))
            try:
                pred = np.argmax(model.predict(img))
            except:
                return json.dumps("Bad image path")

            return dicty[pred]
        return ""

dicty = {0:'blight' ,
         1:'greening',
         2: 'healthy',
         3:'measles',
         4: 'mildew',
         5: 'mold',
         6: 'rot',
         7:'rust',
         8:'scorch',
         9:'spot',
         10:'virus'}

        


api.add_resource(Plant,'/')
api.add_resource(Plant_Disease, '/<string:image_path>')

if __name__ == '__main__':
    app.run(debug=True)


        









