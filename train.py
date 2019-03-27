"""
Created on Sat Mar 16 14:30:20 2019

@author: Antoine Prat
"""

import model_text
import sys
from keras.models import load_model

def train(epoch, retrain):
    
    if retrain == 1:
        #on charge le modele RNN
        model = load_model('RNN_weights/Model.h5')
        model.load_weights('RNN_weights/Weights.h5')
        
        batch_size = 512
        model.fit_generator(descriptor.data_process(batch_size=batch_size),  #fit_generator de modèle text
                    steps_per_epoch=descriptor.no_samples/batch_size, 
                    epochs=epoch, verbose=1, callbacks=None)
        
        model.save('RNN_weights/Model.h5', overwrite=True) #On sauve le modèle
        model.save_weights('RNN_weights/Weights.h5',overwrite=True) #et les poids que nous avons entrainé
    
    elif retrain == 0: 
        descriptor = model_text.model()
        batch_size = 512
        model = descriptor.create_model() #On crée le modèle
        model.fit_generator(descriptor.data_process(batch_size=batch_size),  #fit_generator de modèle text
                            steps_per_epoch=descriptor.no_samples/batch_size, 
                            epochs=epoch, verbose=1, callbacks=None)
        
        model.save('RNN_weights/Model.h5', overwrite=True) #On sauve le modèle
        model.save_weights('RNN_weights/Weights.h5',overwrite=True) #et les poids que nous avons entrainé
    
    else:
        raise ValueError('Mettre 0 ou 1 en second argument, 1 == Retrain, 0 == 1rst train')
 
if __name__=="__main__":
    train(int(sys.argv[1]), int(sys.argv[2]))