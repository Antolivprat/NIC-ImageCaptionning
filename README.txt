Il faut tout d'abord t�l�charger les fichiers TSV � mettre dans data_tsv avec les noms : 

- Train_GCC-training.tsv
- Validation_GCC-1.1.0-Validation.tsv

On peut les t�l�charger ici : https://ai.google.com/research/ConceptualCaptions

Il faut ensuite faire tourner : 
- get_data.py Qui va t�l�charger les donn�es � partir des deux fichiers TSV
- preprocess_data.py Qui va les process pour ne pas avoir d'erreur dans le training
- model_image.py Qui va cr�er l'encoding des donn�es de training
- train.py 50 0 Qui va cr�er notre mod�le et l'entrainer
- test.py Nmb O� Nmb indique le nombre de donn�e que l'on veut predict dans notre set de validation