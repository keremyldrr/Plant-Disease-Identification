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
from flaskext.mysql import MySQL
import urllib
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
app = Flask(__name__)
api = Api(app)
mysql = MySQL()
app.config['MYSQL_DATABASE_USER'] = 'plantid_user'
app.config['MYSQL_DATABASE_PASSWORD'] = 'lvFu3&48rd#s'
app.config['MYSQL_DATABASE_DB'] = 'plantid'
app.config['MYSQL_DATABASE_HOST'] = '10.36.48.64'
mysql.init_app(app)
conn = mysql.connect()
cursor = conn.cursor()
class check(Resource):
    def get(self):
        cursor.execute("SELECT id,path_web FROM disease_job WHERE status=0")
        all_data = cursor.fetchall()
        json_file = open('model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = tf.keras.models.model_from_json(loaded_model_json)
        model.load_weights("weights.h5")
        for data in all_data:
            path = urllib.request.urlopen( "http://plantid.sabanciuniv.edu/"+data[1])
            img = plt.imread(path,format='jpg')#format check, error handling
           
            img = np.resize(img,(256,256,3))
            img = np.expand_dims(img,axis=0)        
            pred = np.argmax(model.predict(img))
            print(dicty[pred])
            query = "UPDATE disease_job SET result='"+dicty[pred]+"',status=2 WHERE id="+str(data[0])
            print(query)
            cursor.execute(query)

        return json.dumps({"success"}),200

api.add_resource(check,"/identify_disease")
app.run(debug=False)
        





        









