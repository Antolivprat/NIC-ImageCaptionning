# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 16:59:42 2019

@author: Antoine Prat
"""

'''
Pour les photos
'''

import os
exist_already = os.listdir('data/Images')
CCP_train = []
CCP_val = []
#On s'occupe maintenant des images qui sont trop petites (ie mal téléchargées)
for i in exist_already:
    if os.path.getsize('data/Images/' + i ) > 10000:
        if i[:5] == 'train':
            CCP_train.append(str(i))
        elif i[:3] == 'val':
            CCP_val.append(str(i))
    else:
        pass
    
print('small images deleted')

from keras.preprocessing import image
import warnings
warnings.filterwarnings('error')
potentials_errors_train = []
for i in range(len(CCP_train)): 
    try:
        a = image.load_img('data/Images/' + CCP_train[i], target_size=(224,224))
    except: #Si l'image ne s'ouvre pas, on considère ça comme une erreur, on va la supprimer de notre dataset
        print(i, CCP_train[i])
        potentials_errors_train.append(i)
        
print('Training images done')

potentials_errors_val = []
for i in range(len(CCP_val)):
    try:
        a = image.load_img('data/Images/' + CCP_val[i], target_size=(224,224))
    except:
        print(i, CCP_val[i])
        potentials_errors_val.append(i)
        
print('Validation images done')

for i in potentials_errors_train[::-1]: #On supprime ce qui n'est pas bon
   del CCP_train[i]
    
for i in potentials_errors_val[::-1]:
    del CCP_val[i]


with open('data/CCP.trainImages.txt', 'w') as f: #On réenregistre
    f.write(CCP_train[0])
    for i in CCP_train[1:]:
        f.write('\n' + i)

with open('data/CCP.valImages.txt', 'w') as f: #On réenregistre
    f.write(CCP_val[0])
    for i in CCP_val[1:]:
        f.write('\n' + i)

'''
Pour les captions
'''
        
caption_file_train = open('data/Text/pre_trainimgs.txt').read().split('\n')
caption_file_val = open('data/Text/pre_valimgs.txt').read().split('\n')
CCP_train = open('data/CCP.trainImages.txt').read().split('\n')
CCP_val = open('data/CCP.valImages.txt').read().split('\n')

caption_file_train[0] = caption_file_train[0][3:]
caption_file_train_final = []
caption_file_val_final = []
#On arrange la dataframe pour enlever les , qui apparaissent dans les phrases !
#On enlève aussi tous les caractères spéciaux

#On ne prends pas les données qu'on a enlevé dans notre étape précédente
for i in range(len(caption_file_train)):
    if caption_file_train[i].split('<start>')[0].replace('\'', '')[:-2] in CCP_train:
        caption_file_train_final.append(caption_file_train[i].split('<start>')[0].replace('\'', '') + '<start>' + caption_file_train[i].split('\'<start>')[-1].replace(', ', '').replace('# ', '').replace('-', '').replace(':', '').replace(';', '').replace('...', '').replace(',', '').replace('\'', '').replace('? ', '').replace('! ', '').replace('. ', '').replace('  ', ''))
    
for i in range(len(caption_file_val)):
    if caption_file_val[i].split('<start>')[0].replace('\'', '')[:-2] in CCP_val:
        caption_file_val_final.append(caption_file_val[i].split('<start>')[0].replace('\'', '') + '<start>' + caption_file_val[i].split('\'<start>')[-1].replace(', ', '').replace('# ', '').replace('-', '').replace(':', '').replace(';', '').replace('...', '').replace(',', '').replace('\'', '').replace('? ', '').replace('! ', '').replace('. ', '').replace('  ', ''))


with open('data/Text/trainimgs.txt', 'w') as f: #On réenregistre
    f.write(caption_file_train_final[0])
    for i in caption_file_train_final[1:]:
        f.write('\n' + i)
        
with open('data/Text/valimgs.txt', 'w') as f: #On réenregistre
    f.write(caption_file_val_final[0])
    for i in caption_file_val_final[1:]:
        f.write('\n' + i)