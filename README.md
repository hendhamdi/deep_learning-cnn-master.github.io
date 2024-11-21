# Classification d'Images avec un Réseau Neuronal Convolutif (CNN)

## Description
Ce projet utilise PyTorch pour créer, entraîner et évaluer un réseau neuronal convolutif (CNN) destiné à la classification d'images. Les données sont divisées en ensembles d'entraînement (80%) et de test (20%), et des métriques telles que la perte (loss) et la précision (accuracy) sont suivies pour analyser les performances du modèle.

## Fonctionnalités
 **Partitionnement des données** : Division du dataset en ensembles d'entraînement (80%) et de test (20%) à l'aide du script `split.py`.
- **Modèle CNN** : Construction d’un réseau avec des couches convolutionnelles, de normalisation, d’activation (*ReLU*), de *pooling*, et une couche entièrement connectée pour la classification.
- **Optimisation avancée** : Utilisation de l’algorithme SGD (*Stochastic Gradient Descent*), avec ajustement des hyperparamètres tels que le taux d’apprentissage et le *momentum*.
- **Analyse des performances** : Suivi des métriques au fil des époques, incluant la perte et la précision.
- **Visualisation** : Génération d’un graphique illustrant la perte et la précision au fil des époques, sauvegardé au format PDF.

## Dataset
Le dataset utilisé contient les classes suivantes :
- Annual Crop
- Forest
- River
- Sea Lake
- Highway
- Industrial
- Pasture
- Permanent Crop
- Residential
- Herbaceous Vegetation

## Dépendances
Assurez-vous d'avoir les bibliothèques suivantes installées :
- Python 3.x
- PyTorch
- Matplotlib
- Scikit-learn
- MySQL (si besoin de stockage supplémentaire)

## Utilisation
### Partitionnement des données
Le script `split.py` dans le répertoire `other` divise le dataset en ensembles d'entraînement et de test :
    ```python
    from sklearn.model_selection import train_test_split
    
Exemple d'utilisation inclus dans le script

### Entraînement du modèle
.Phase d'entraînement : Ajustement des poids avec rétropropagation.

.Phase de test : Évaluation de la capacité du modèle à généraliser.

## Résultats
.Précision finale : 85% (entraînement), 81% (test).

.Tendances observées : Diminution progressive de la perte, augmentation constante de la précision.

## Visualisation
Un graphique illustrant la perte et la précision par époque est généré et sauvegardé en PDF.

![image](https://github.com/user-attachments/assets/fc3eea4d-0a4c-419a-98e6-85de635e511e)
![image](https://github.com/user-attachments/assets/202f89dd-5432-4479-bf0b-e5b6c99158a6)

