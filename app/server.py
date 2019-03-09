import os
import numpy as np
import pprint
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from flask import Flask
from flask_restful import reqparse, abort, Api, Resource
app = Flask(__name__)
api = Api(app)
class Plant_Disease(Resource):
    def get(self,image_path=""):
        print(image_path)
        if image_path!="favicon.ico":
            img = plt.imread(image_path[:])
            img = np.reshape(img,[224,224,3])
            img = np.expand_dims(img,axis=0)
        
            try:
                pred = np.argmax(model.predict(img))
            except:
                return "Bad image path"
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

        
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = tf.keras.models.model_from_json(loaded_model_json)
model.load_weights("weights.h5")

api.add_resource(Plant_Disease, '/<string:image_path>')

if __name__ == '__main__':
    app.run(debug=True)


        









