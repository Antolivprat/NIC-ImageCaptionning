# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 14:29:29 2019

@author: Antoine Prat
"""

"""
On crée depuis les fichiers tsv train et val deux fichiers textes.
Le fichier CCP.train.Images qui contient le nom des images
Le fichier trainimg qui contient à chaque ligne (nom_img, <start> caption <end>)
"""


"""
Pour les données de training
"""
import pandas as pd
import numpy as np
import urllib.request
import requests #faster than urllib, we will use for training
import os
import re

exist_already = os.listdir('data/Images') 

train_data = pd.read_csv('data_tsv/Train_GCC-training.tsv', delimiter='\\t', header=None)
CCP = []
caption_file = []
TIMEOUT=5

for i in range(len(train_data[:200000])):
    name = "train_" + str(i) +  ".jpg"
    caption = '<start> ' + train_data.iloc[i, 0] + ' <end>' #On veut les captions de la forme "<start> ... <end>'
    url = train_data.iloc[i, 1]
    
    if name in exist_already: #On ne retélécharge pas les déjà faites
        CCP.append(name) 
        caption_file.append([name, caption])
        
    else:
        try:
            r = requests.get(url, timeout=TIMEOUT)
            if 'image' in r.headers['content-type']:
                with open("data/Images/" + name, 'wb') as f:  
                    f.write(r.content)
                CCP.append(name) #Fichier avec les noms
                caption_file.append([name, caption]) #Fichier nom / caption
            else:
                raise ValueError 
                
        except KeyboardInterrupt:
            raise
            
        except: #S'il y a une erreur, on continue sans rajouter le fichier, (Image qui n'existe pas, ou mauvais site)
            print(i)
            pass
    
with open('data/CCP.trainImages.txt', 'w') as f:
    f.write(CCP[0])
    for i in CCP[1:]:
        f.write('\n' + i)
        
with open('data/Text/pre_trainimgs.txt', 'w') as f:
    f.write('\'' + caption_file[0][0] + '\'' + ', ' + '\'' + caption_file[0][1].replace(', ', '') + '\'')
    for i in caption_file[1:]:
        f.write('\n' + '\'' + i[0] + '\'' + ', ' + '\'' + i[1] + '\'')


"""
Pour les données de validations/test
"""
val_data = pd.read_csv('data_tsv/Validation_GCC-1.1.0-Validation.tsv', delimiter='\\t', header=None)
CCP = []
caption_file = []

for i in range(len(val_data)):
    name = "val_" + str(i) +  ".png"
    caption = '<start> ' + val_data.iloc[i, 0] + ' <end>'
    url = val_data.iloc[i, 1]
    
    try:
        urllib.request.urlretrieve(url, "data/Images/" + name) #On enregistre l'image
        CCP.append(name) #Fichier avec les noms
        caption_file.append([name, caption]) #Fichier nom / caption
    except KeyboardInterrupt:
        raise
        
    except: 
        print(i)
        pass
    
with open('data/CCP.valImages.txt', 'w') as f: #1er fichier
    f.write(CCP[0])
    for i in CCP[1:]:
        f.write('\n' + i)
        
with open('data/Text/pre_valimgs.txt', 'w') as f: #2nd fichier
    f.write('\'' + caption_file[0][0] + '\'' + ', ' + '\'' + caption_file[0][1].replace(', ', '') + '\'')
    for i in caption_file[1:]:
        f.write('\n' + '\'' + i[0] + '\'' + ', ' + '\'' + i[1] + '\'')
        
    
    

