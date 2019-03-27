Il faut tout d'abord télécharger les fichiers TSV à mettre dans data_tsv avec les noms : 

- Train_GCC-training.tsv
- Validation_GCC-1.1.0-Validation.tsv

On peut les télécharger ici : https://ai.google.com/research/ConceptualCaptions

Il faut ensuite faire tourner : 
- get_data.py Qui va télécharger les données à partir des deux fichiers TSV
- preprocess_data.py Qui va les process pour ne pas avoir d'erreur dans le training
- model_image.py Qui va créer l'encoding des données de training
- train.py 50 0 Qui va créer notre modèle et l'entrainer
- test.py Nmb Où Nmb indique le nombre de donnée que l'on veut predict dans notre set de validation