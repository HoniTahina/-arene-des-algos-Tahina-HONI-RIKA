# 🥊 Arène des Algos — J3

Benchmark comparatif de modèles de Machine Learning sur quatre problèmes réels, culminant sur un **Fight des IA** avec leaderboard.

Auteur : Tahina HONI-RIKA  
Repo : [arene-des-algos-Tahina-HONI-RIKA](https://github.com/HoniTahina/-arene-des-algos-Tahina-HONI-RIKA)

---

## Structure du projet

```
j3/
├── arene_des_algos_J3.ipynb   # Notebook principal (une section par phase)
├── listings.csv               # Dataset AirBnB (Inside Airbnb)
├── SMSSpamCollection.txt      # Dataset SMS Spam (UCI)
├── sonar.csv                  # Dataset Sonar Mines/Rochers (UCI)
└── README.md                  # Ce fichier
```

---

## Phases

| Phase | Problème | Dataset | Type | Algos |
|-------|----------|---------|------|-------|
| A | Prix immobiliers | California Housing (sklearn) | Régression | LinearRegression, RandomForest |
| B | Segmentation AirBnB | listings.csv (Inside Airbnb) | Clustering | KMeans |
| C | Spam vs normal | SMSSpamCollection.txt (UCI) | Classification texte | MultinomialNB, LogisticRegression |
| D | Mine vs rocher | sonar.csv — 208 lignes, 60 vars | Classification binaire | LogisticRegression, SVC rbf, RandomForest |
| E | Fight des IA | Sonar (même split que Phase D) | Benchmark multi-algos | 6 algorithmes |

---

## Phase A — Régression immobilière (California Housing)

### Résultats cas normal (20 640 lignes, random_state=42)

| Modèle | R² | MAE | RMSE |
|--------|----|-----|------|
| LinearRegression | 0.576 | 0.533 | 0.746 |
| RandomForest | 0.805 | 0.327 | 0.505 |

### Réponses aux checkpoints

**Cas limite — 100 lignes seulement :**  
Contre-intuitivement, les métriques *semblent* meilleures sur 100 lignes (LR : R²=0.71, RF : R²=0.81). C'est un **artefact statistique** : avec si peu de données, le modèle a plus de chances de tomber sur un split favorable par hasard. En pratique, un modèle entraîné sur 100 points ne généralise pas — il mémorise. Pour le vérifier, il faudrait tester sur un jeu de données complètement indépendant : le R² s'effondrerait.

**Cas adversarial — revenu médian=0, population=9000 :**  
Le Random Forest prédit **2.70** (270 000 $). Cette valeur n'est pas absurde en apparence, mais elle est **extrapolée hors de la distribution d'entraînement** : un quartier à revenu zéro avec 9000 habitants n'existe presque pas dans le jeu d'entraînement. En production, il faut ajouter une **vérification des bornes** (ex. MedInc ∈ [0.5, 15]) et rejeter ou signaler les entrées hors plage plutôt que de leur faire confiance aveuglément.

---

## Phase B — Clustering AirBnB

**Dataset :** listings.csv — 7 194 lignes conservées après nettoyage (dropna sur 7 colonnes numériques).  
**Colonnes retenues :** price, minimum_nights, number_of_reviews, reviews_per_month, availability_365, number_of_reviews_ltm, calculated_host_listings_count.

### Choix de k (après standardisation)

| k | Inertie | Silhouette |
|---|---------|------------|
| 2 | 1 015 335 066 | **0.9945** |
| 3 | 334 763 168 | 0.9745 |
| 4 | 247 901 188 | 0.3426 |
| 5 | 183 583 948 | 0.3901 |
| 6 | 150 735 932 | 0.4046 |
| 7 | 131 706 465 | 0.3892 |
| 8 | 113 197 761 | 0.4024 |

**k=2 retenu** (silhouette maximale = 0.9945).

### Profil des clusters (k=2)

| Cluster | Prix moyen | Nuits min | Nb avis | Avis/mois | Dispo 365 | Description |
|---------|-----------|-----------|---------|-----------|-----------|-------------|
| 0 | 127 € | 2 nuits | 237 avis | 4.41 | 229 j | **Actifs, populaires, prix abordable** — hôtes multi-listings, forte rotation, très disponibles |
| 1 | 185 € | 18 nuits | 36 avis | 1.06 | 197 j | **Séjours longue durée, premium** — moins d'avis, nuits min élevées, prix plus haut |

### Réponses aux checkpoints

**Cas limite — sans standardisation (choisir_k sur df brut) :**  
La colonne `price` varie de 0 à 50 000 € alors que `reviews_per_month` varie de 0 à 58. Sans standardisation, KMeans minimise l'inertie en termes de prix uniquement : les clusters séparent "cheap vs cher" et ignorent toutes les autres dimensions. La silhouette reste élevée mais ne mesure plus des segments sémantiquement utiles — elle mesure juste la séparation sur le prix.

**Cas adversarial — annonce à 50 000 €/nuit :**  
L'annonce à 50 000 € existait déjà dans le dataset (lignes extrêmes repérées avec `idxmax()`). Son effet : le centroïde du cluster "premium" est tiré vers des valeurs aberrantes, rendant les clusters moins interprétables. C'est exactement pourquoi le nettoyage des outliers (ex. `df[df['price'] < 1000]`) est un prérequis absolu avant KMeans : un seul point extrême peut déplacer tout un centroïde.

---

## Phase C — Détection spam (SMS Spam Collection)

**Split :** 80/20, random_state=42, stratifié (pour respecter le déséquilibre 87% ham / 13% spam).  
**Vectorisation :** TF-IDF (pondère les mots rares discriminants).

### Résultats

| Modèle | Precision spam | Recall spam | F1 spam | Accuracy |
|--------|---------------|-------------|---------|----------|
| MultinomialNB | 1.00 | 0.70 | 0.83 | 0.96 |
| LogisticRegression | 1.00 | 0.80 | 0.89 | 0.97 |

**LogisticRegression est meilleure** : recall spam 0.80 vs 0.70 pour NB, avec la même précision parfaite.

### Réponses aux checkpoints

**Happy path :**  
Les deux modèles détectent bien "WINNER!! You have won..." et "Free entry in a cash prize..." comme spam. NB est légèrement plus agressif, LR plus prudent ("URGENT! You are selected for a £1000 reward" → classé ham par LR, spam par NB).

**Edge case — message vide `""` :**  
Le vectorizer TF-IDF ne plante pas sur une chaîne vide — il produit un vecteur tout-zéros. Les deux modèles prédisent **ham** par défaut. Ce comportement est raisonnable (un message vide n'est pas un spam), mais en production il faut détecter et rejeter les messages vides en amont plutôt que de les envoyer au modèle.

**Adversarial — spam déguisé `"salut, ton colis t attend, confirme ici"` :**  
Les deux modèles classent ce message **ham**. Le modèle se fait avoir car ce message utilise un vocabulaire banal absent du corpus d'entraînement anglais (mots français courants, pas de trigger words comme "FREE", "WINNER", "URGENT"). Cela illustre un point critique : **recall et precision racontent deux histoires différentes**. Ici precision spam = 1.00 (zéro faux positif — on ne colle jamais l'étiquette spam à un vrai message), mais recall spam = 0.70-0.80 (on rate 20-30% des spams réels). En pratique, rater un vrai mail important (faux positif spam) est souvent jugé pire que laisser passer un spam — d'où la priorité donnée à la precision.

---

## Phase D — Sonar Mines vs Rochers

**Dataset :** 208 échantillons, 60 variables de fréquence, classes : M=mine (111), R=rocher (97).  
**Split :** 80/20, random_state=42, stratifié.

### Résultats avec standardisation

| Modèle | Precision mine | Recall mine | F1 mine | Accuracy |
|--------|---------------|-------------|---------|----------|
| LogisticRegression | 0.83 | 0.86 | 0.84 | 0.83 |
| SVC rbf | **0.88** | **1.00** | **0.94** | **0.93** |
| RandomForest | 0.79 | 0.86 | 0.83 | 0.81 |

### Résultats sans standardisation (cas limite)

| Modèle | F1 mine | Accuracy | Δ vs standardisé |
|--------|---------|----------|------------------|
| LogisticRegression | 0.83 | 0.81 | ≈ stable* |
| SVC rbf | 0.86 | 0.83 | −0.08 F1 |
| RandomForest | 0.83 | 0.81 | ≈ stable |

*La régression logistique sonar reste stable car les 60 variables sont déjà sur des échelles similaires (valeurs entre 0 et 1). Le SVC en revanche chute significativement : le kernel RBF calcule des distances euclidiennes, sensibles à l'échelle.

### Réponses aux checkpoints

**Cas limite — sans standardisation :**  
Le SVC rbf perd environ 0.08 de F1 sans standardisation. La régression logistique et le Random Forest sont peu affectés sur ce dataset particulier car les 60 fréquences du sonar sont déjà normalisées entre 0 et 1 (nature du signal). En règle générale : **toujours standardiser avant SVM**, même si les données semblent bien échelonnées.

**Cas adversarial — capteur en panne (60 valeurs = 0) :**  
Le modèle produit quand même une prédiction avec une certaine confiance. C'est dangereux : un vecteur tout-zéros est hors de toute distribution réelle d'entraînement. En contexte militaire/sécurité, il faudrait ajouter un **détecteur d'anomalies en amont** (ex. vérifier que la norme L2 du vecteur dépasse un seuil minimal) et refuser de prédire si le signal est invalide — plutôt qu'afficher une fausse certitude.

---

## Phase E — Fight des IA

**Dataset :** Sonar (même split que Phase D — `random_state=42`, stratifié)  
**Métrique :** F1-score classe Mine (1) — car rater une mine est plus coûteux qu'une fausse alarme  
**Note importante :** le Fight a tourné sur le split de Phase D qui utilisait les dernières valeurs de `X_train/X_test` en mémoire (données non standardisées pour les modèles sans pipeline intégré).

### Leaderboard final

| Rang | Algorithme | F1 (Mine) | Temps entraînement |
|------|------------|-----------|-------------------|
| 🥇 1 | GradientBoosting | **0.864** | 1.53 s |
| 🥈 2 | DecisionTree | 0.857 | 0.02 s |
| 🥈 2 | SVC_rbf | 0.857 | 0.02 s |
| 4 | RandomForest | 0.844 | 1.41 s |
| 5 | LogisticRegression | 0.826 | 0.02 s |
| 6 | NaiveBayes | 0.732 | 0.01 s |

---

## 🏆 Champion : GradientBoosting

### Pourquoi le F1 et pas l'accuracy ?

Le dataset Sonar est quasi-équilibré (111 mines / 97 rochers), donc l'accuracy serait acceptable. Le **F1-score** est néanmoins retenu car il pénalise simultanément les faux positifs et les faux négatifs. Dans un contexte sécurité, rater une vraie mine (faux négatif) a des conséquences bien plus graves qu'une fausse alarme — on pourrait même argumenter pour maximiser le **recall** seul. Le F1 est un bon compromis équilibré.

### Pourquoi GradientBoosting ?

GradientBoosting obtient le meilleur F1 (0.864) sur ce jeu de test. Il construit des arbres séquentiellement en corrigeant les erreurs du modèle précédent, ce qui lui permet de capturer des interactions complexes entre les 60 fréquences du sonar. Sur un dataset aussi petit (208 lignes), il s'en sort mieux que RandomForest (0.844) car il optimise directement la loss à chaque étape plutôt que de miser sur la variance des arbres.

### Pourquoi pas DecisionTree ou SVC_rbf, à égalité en 2e position ?

DecisionTree (F1=0.857, 0.02s) est remarquablement rapide mais instable : un seul arbre sur 166 points d'entraînement est sujet à l'overfitting. Sans validation croisée, son score ici peut être dû à un découpage chanceux.

SVC rbf (F1=0.857) est très compétitif et c'est son terrain de prédilection (peu de données, beaucoup de variables). La différence avec GradientBoosting est marginale (0.007 de F1). En production, si le temps d'entraînement compte (modèle réentraîné fréquemment), SVC rbf à 0.02s serait préférable à GradientBoosting à 1.53s.

### Verdict arbitrage score / vitesse

| Scénario | Champion recommandé | Raison |
|----------|---------------------|--------|
| Précision maximale, batch | GradientBoosting | Meilleur F1, temps acceptable en entraînement offline |
| API temps réel, réentraînement fréquent | SVC rbf | F1 quasi-identique, 75× plus rapide à entraîner |
| Interprétabilité requise | LogisticRegression | Coefficients lisibles, F1 correct (0.826) |

---

## Reproduire les résultats

```bash
# Dépendances
pip install scikit-learn pandas numpy matplotlib

# Fichiers nécessaires (à placer dans le même dossier que le notebook)
# - listings.csv       (Inside Airbnb, ex. Paris)
# - SMSSpamCollection.txt  (UCI SMS Spam Collection)
# - sonar.csv          (UCI Connectionist Bench)

# Lancer le notebook
jupyter notebook arene_des_algos_J3.ipynb
```

Tous les splits utilisent `random_state=42`. Pour reproduire exactement le leaderboard : `fight_des_ia(X_train, X_test, y_train, y_test, metrique=f1_mines)`.

---

## Stratégie de commits

```
git add j3/arene_des_algos_J3.ipynb
git commit -m "Phase A : California Housing — LinearReg R2=0.58, RF R2=0.81"
git commit -m "Phase A : checkpoints limite (100 lignes) et adversarial (revenu=0)"
git commit -m "Phase B : clustering AirBnB — k=2 retenu, silhouette=0.99"
git commit -m "Phase B : checkpoints sans standardisation et outlier 50000€"
git commit -m "Phase C : spam TF-IDF — NB recall=0.70, LR recall=0.80"
git commit -m "Phase C : checkpoints message vide et spam déguisé"
git commit -m "Phase D : sonar — SVC F1=0.94, LR F1=0.84, RF F1=0.83"
git commit -m "Phase D : checkpoints sans standardisation et capteur en panne"
git commit -m "Phase E : Fight des IA — leaderboard final, champion GradientBoosting F1=0.864"
git commit -m "README : champion justifié, réponses aux checkpoints, leaderboard réel"
```
