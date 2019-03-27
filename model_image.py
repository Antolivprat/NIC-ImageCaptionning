# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 21:55:23 2019

@author: Antoine Prat
"""

"""
In this code, we encode thanks to the model VGG our images 
and load it in Output_step1/image_encodings.p. We only encode the training
and validation images

"""

from vgg16 import VGG16  #CNN qui va nous donner l'encoding
import numpy as np
from keras.preprocessing import image
from imagenet_utils import preprocess_input      #preprocessing pour que les images aient la même taille
import six.moves.cPickle as pickle
import progressbar

def model_gen():#crée le modèle avec les poids de imagenet
    model = VGG16(weights='imagenet', include_top=True, input_shape = (224, 224, 3))
    return model

def encodings(model, path): #encoding d'une image
    processed_img = image.load_img(path, target_size=(224,224))
    x = image.img_to_array(processed_img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    image_final = np.asarray(x)
    prediction = model.predict(image_final)
    prediction = np.reshape(prediction, prediction.shape[1])
    return prediction

def encode_image(): #crée l'encoding des images trains et output un pickle "Data/CCP.trainImages.txt"
    model = VGG16(weights='imagenet', include_top=True, input_shape = (224, 224, 3))
    image_encodings = {}
    
    train_imgs_id = open("data/CCP.trainImages.txt").read().split('\n') #La liste contenant le nom des images
    images = []
    images.extend(train_imgs_id)
    #progress bar
    bar = progressbar.ProgressBar(maxval=len(images), widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    counter=1
    print("Encoding images")
    
    for img in images: #On encode les images dans notre fichier
        path = "data/Images/" + str(img)
        image_encodings[img] = encodings(model, path) #image_encodings est un dictionnaire
        bar.update(counter)
        counter += 1

    bar.finish()
    with open("Image_encoding/image_encodings.p", "wb") as pickle_f: #On enregistre le pickle de nos images encoded
        pickle.dump(image_encodings, pickle_f)
    print("Encoding done")

if __name__=="__main__":
    encode_image()