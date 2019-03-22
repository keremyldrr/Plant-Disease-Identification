import os
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.vis_utils import plot_model

'''
from google.colab import drive
drive.mount('/content/drive')

#Github access
!pip install -q xlrd
!git clone https://github.com/keremyldrr/Plant-Disease-Identification.git

# Files from the cloned git repository.
!ls Plant-Disease-Identification/12CLASS/train
!ls Plant-Disease-Identification/12CLASS/test
'''

#Train Data Generator with Data Augmentation(Flips, Normalizations, etc.), Train/Validation = 0.8/0.2
train_gen = ImageDataGenerator(featurewise_center=True, samplewise_center=True, 
                               featurewise_std_normalization=True,
                               samplewise_std_normalization=True,
                               zca_whitening=False, zca_epsilon=1e-06,
                               rotation_range=0,
                               width_shift_range=0.0,
                               height_shift_range=0.0, 
                               brightness_range=None,
                               shear_range=0.0, 
                               zoom_range=0.0, channel_shift_range=0.0, fill_mode='nearest', cval=0.0, horizontal_flip=True,
                               vertical_flip=True, rescale=None, preprocessing_function=None, data_format=None,validation_split=None, dtype=None)

#Preprocessing function
#random_rotation(x, 45)

#Train Data Generator with Data Augmentation(Flips, Normalizations, etc.), Train/Validation = 0.8/0.2
validation_gen = ImageDataGenerator(featurewise_center=True, samplewise_center=True, 
                               featurewise_std_normalization=True,
                               samplewise_std_normalization=True,
                               zca_whitening=False, zca_epsilon=1e-06,
                               rotation_range=0,
                               width_shift_range=0.0,
                               height_shift_range=0.0, 
                               brightness_range=None,
                               shear_range=0.0, 
                               zoom_range=0.0, channel_shift_range=0.0, fill_mode='nearest', cval=0.0, horizontal_flip=True,
                               vertical_flip=True, rescale=None, preprocessing_function=None, data_format=None,validation_split=None, dtype=None)

#Preprocessing function
#random_rotation(x, 45)

#Test Data Generator
test_gen = ImageDataGenerator(featurewise_center=True, samplewise_center=True, 
                               featurewise_std_normalization=True,
                               samplewise_std_normalization=True,
                               zca_whitening=False, zca_epsilon=1e-06,
                               rotation_range=0,
                               width_shift_range=0.0,
                               height_shift_range=0.0, 
                               brightness_range=None,
                               shear_range=0.0, 
                               zoom_range=0.0, channel_shift_range=0.0, fill_mode='nearest', cval=0.0, horizontal_flip=True,
                               vertical_flip=True, rescale=None, preprocessing_function=None, data_format=None,validation_split=None, dtype=None)

"""#12 CLASS DATA"""

#For 12 class
image_data = train_gen.flow_from_directory("/cta/users/barissevilmis/workfolder/Plant-Disease-Identification/train", shuffle = True, target_size = (224, 224), class_mode = 'binary')
#print(image_data)

validation_data = validation_gen.flow_from_directory("/cta/users/barissevilmis/workfolder/Plant-Disease-Identification/validation", shuffle = True, target_size = (224, 224), class_mode = 'binary')
#print(validation_data)

test_data = test_gen.flow_from_directory("/cta/users/barissevilmis/workfolder/Plant-Disease-Identification/test", shuffle = False, target_size = (224, 224), class_mode = 'binary')
#print(test_data)

"""#12 CLASS ONLY LAST THREE LAYERS TRAINABLE"""

#For 12 classes  -> training_12_2.cpkt
#Load pretrained(imagenet) VGG-16 model, leave out FC and Softmax layers
vgg16 = tf.keras.applications.VGG16(include_top = False, weights = 'imagenet', pooling = 'max')

#Only make last three layer trainable
for i in range(17):
  vgg16.layers[i].trainable = False

model = tf.keras.models.Sequential()
model.add(vgg16)
model.add(tf.keras.layers.Dense(12 ,activation = 'softmax'))

#See the model summary
#vgg16.summary()
model.summary()

#12 Classes: CASE 2
checkpoint_path = "/cta/users/barissevilmis/workfolder/weights"

checkpoint_dir = os.path.dirname(checkpoint_path)

# Create checkpoint callback
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, 
                                                 monitor='val_loss',
                                                 save_weights_only=True,
                                                 save_best_only=True,
                                                 verbose = 1)
csv_logger = tf.keras.callbacks.CSVLogger("training.log")
                                                 
"""#COMPILE THE MODEL"""

model.compile(
    optimizer=tf.train.AdamOptimizer(learning_rate=1e-3, ),
    loss=tf.keras.losses.sparse_categorical_crossentropy,
    metrics=['acc']
)

#Load pretrained weights
#model.load_weights(checkpoint_path)
print("Training has started!!")
#SKIP this block, if you only want to evaluate the results
#Fit generator
model.fit_generator(
    image_data,
    epochs=10,
    steps_per_epoch=200,
    validation_data = validation_data,
    callbacks = [cp_callback, csv_logger],
    verbose = 1  
)
print("Training has ended!!")
"""#TEST"""

#Evaluate performance over test data
model.load_weights(checkpoint_path)
loss,acc = model.evaluate_generator(test_data)
print("Loss(Test Data)", loss)
print("Accuracy(Test Data)", acc)
#See class probabilities
pred = model.predict_generator(test_data)
print("Test has ended!!!")
