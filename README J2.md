# Arène des Algorithmes IA

## Objectif

Ce projet compare plusieurs algorithmes de machine learning sur différents datasets de classification afin de comprendre :

- leurs performances
- leurs comportements
- leurs limites
- l’impact du prétraitement des données

L’objectif n’est pas uniquement de maximiser l’accuracy, mais de comprendre **quel modèle est adapté à quel type de problème**.

---

## Datasets utilisés

### 1. Breast Cancer (binaire)
- 2 classes : bénin / malin
- Données médicales
- 30 features numériques

### 2. Wine (multi-classe)
- 3 classes : types de vin
- 13 features chimiques

---

## Algorithmes testés

- Logistic Regression
- Decision Tree
- K-Nearest Neighbors (KNN)

---

## Pipeline de traitement

Chaque expérimentation suit le même pipeline :

1. Chargement du dataset
2. Séparation train / test
3. (Optionnel) normalisation des données
4. Entraînement des modèles
5. Prédiction
6. Évaluation (accuracy)
7. Comparaison des résultats

Le test set n’est jamais utilisé pendant l’entraînement (pas de data leakage).

---

## Résultats globaux (synthèse)

### Breast Cancer

| Modèle | Performance | Observation |
|--------|------------|-------------|
| Logistic Regression | Très élevée | Stable et fiable |
| KNN | Élevée | Très sensible au scaling |
| Decision Tree | Bonne | Interprétable mais instable |

---

### Wine Dataset

| Modèle | Performance | Observation |
|--------|------------|-------------|
| Logistic Regression | Très bonne | Bon modèle généraliste |
| KNN | Bonne | dépend fortement des distances |
| Decision Tree | Moyenne | tendance au sur-apprentissage |

---

## Modèle retenu

 **Logistic Regression**

---

## Pourquoi ce choix ?

- Bon compromis performance / simplicité
- Très stable sur différents datasets
- Facile à interpréter
- Fonctionne bien en classification binaire et multi-classe

---

## Limites observées

- KNN dépend fortement de l’échelle des données
- Decision Tree peut sur-apprendre facilement
- Les résultats peuvent être biaisés en cas de data leakage
- L’accuracy seule ne suffit pas à évaluer un modèle

---

## Impact du scaling

- KNN et Logistic Regression sont fortement améliorés par la normalisation
- Decision Tree est peu affecté

---

## Conclusion

Ce projet montre que :

- Il n’existe pas un modèle universel
- Le choix dépend des données
- Le prétraitement est aussi important que l’algorithme
- L’analyse des erreurs est essentielle (matrice de confusion)

