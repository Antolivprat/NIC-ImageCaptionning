"""
Created on Sat Mar 16 14:31:05 2019

@author: Antoine Prat
"""

"""
Input : Chemin a un fichier .txt (exemple.txt) qui contient le nom de tous photos
        que nous souhaitons prédire

Output : Crée un fichier .txt (exemple_pred.txt) qui contient les tuples 
        (nom fichier, annotation).
        
*****************************************************

Se sert des modèles dans model_image.py et model_text.py.
Le premier a été entrainé sur imagenet et le second avec train.py

Nous faisons un post_processing des prédiction dans post_process.py
"""
import model_image
import model_text
import test_postprocess as pp
import sys
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu

def text(path, imgs):
    #On charge le modèle VGG
    encode = model_image.model_gen()
    
    #on charge le modèle RNN
    descriptor = model_text.model()
    model = descriptor.create_model(ret_model = True)
    
    #On charge les poids qui vont avec
    weight = 'RNN_weights/Weights.h5'
    model.load_weights(weight)
        
    if isinstance(imgs, list): #si nous avons une liste d'imagess
        encoded_images = [(img, model_image.encodings(encode, path + img)) for img in imgs]
        image_captions = [(img, pp.generate_captions(descriptor, model, encoding, beam_size=3)) for img, encoding in encoded_images]
        return image_captions
        
    elif isinstance(imgs, str): #Si nous avons une image unique
        image_path = path + imgs
        encoded_image = model_image.encodings(encode, image_path)
        image_captions = (imgs, pp.generate_captions(descriptor, model, encoded_image, beam_size=3))
        return image_captions

    else:
        raise ValueError('Please change function test() to give the right location and list of images to predict')
        
    print(image_captions)
    
def test(number):
    path = "data/Images/"  #Où se situent nos images
    test_imgs_ids = open("data/CCP.valImages.txt").read().split('\n') #La liste contenant le nom des images
    
    captions = text(path, test_imgs_ids[:number])
    dataframe = pd.read_csv('data/Text/valimgs.txt', delimiter=',', header=None)
       
    for i in captions:
        print('Pour l\'image : ', i[0])
        bleu = sentence_bleu(str(dataframe[dataframe[0]==i[0]][1].values)[11:-8].split(), i[1].split())
        print('Nous avons un score de : ', bleu)


    with open('data/Text/predictionvals.txt', 'w') as f:
        f.write(str(captions[0]).replace('(', '').replace(')',''))
        for i in range(1, len(captions)):
            f.write('\n' + str(captions[i]).replace('(', '').replace(')',''))

if __name__ == '__main__':
	test(int(sys.argv[1]))
