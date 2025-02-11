# rnd-multimodality
Projet R&amp;D multimodalité 2024

Ce projet a pour objet de produire un modèle transformer (type GPT2 ) qui soit capable d'ingérer à la fois des images, mais aussi des séries temporelles.
Le modèle doit être capable de produire des prédictions même lorsque l'un des types de données est manquant.

Le système d'ingestion de ces données a été codé(dataloader). Les images sont mises au bon format (L1) via un auto encodeur variationnel à vecteurs quantisés. Les séries temporelles sont mises au bon format via un modèle Chronos qui quantifie ces données via un nombre fini de bins (L2). 
Le vocabulaire de notre modèle est donc la concaténation des tokebs L1+ L2.
