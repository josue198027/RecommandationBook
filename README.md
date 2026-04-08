# RecommandationBook
# BookRecommandation

Systeme intelligent de recommandation de livres base sur le dataset [Book Recommender System (Item-Based)](https://www.kaggle.com/datasets/thedevastator/book-recommender-system-itembased) de Kaggle.

Le projet combine cinq moteurs de recommandation (Baseline, Content-Based, Collaborative Filtering, Matrix Factorization et Hybride) pour suggerer des livres personnalises a chaque utilisateur. Une application web interactive permet de visualiser les resultats et de comparer les performances de chaque approche.

---

## Donnees du projet

Le dataset est compose de 5 fichiers CSV :

| Fichier | Lignes | Description |
|---|---|---|
| `collaborative_books_df.csv` | 196 296 | Table principale : interactions utilisateur-livre avec notes (1 a 5) |
| `collaborative_book_metadata.csv` | 96 | Metadonnees enrichies (description, genre, auteur, nb pages) |
| `book_id_map.csv` | 2 360 650 | Mapping `book_id_csv` vers `book_id` original |
| `user_id_map.csv` | 876 145 | Mapping `user_id_csv` vers UUID utilisateur |
| `book_titles.csv` | 1 447 341 | Dictionnaire `book_id` vers titre complet |

### Chiffres cles

- **66 909** utilisateurs uniques
- **898** livres uniques
- **196 296** interactions (evaluations)
- **Sparsite** : 99.67% (la matrice user x book est tres creuse)
- **Note moyenne** : 3.923 / 5
- **Cold-start** : 81.7% des utilisateurs ont moins de 5 notes
- **Couverture metadata** : seulement 95 des 898 livres (10.6%) disposent de metadonnees

### Distribution des notes

| Note | Nombre | Proportion |
|---|---|---|
| 1 | 4 837 | 2.5% |
| 2 | 12 303 | 6.3% |
| 3 | 42 680 | 21.7% |
| 4 | 69 813 | 35.6% |
| 5 | 66 663 | 34.0% |

Le dataset presente un biais positif marque : 69.5% des notes sont 4 ou 5 (ratio hauts/bas de 7.96x).

---

## Architecture du pipeline

Le notebook `Recommandation_de_livre_(1).ipynb` suit un pipeline en 10 cellules :

### 1. Ingestion des donnees
Chargement des 5 fichiers CSV avec gestion des types (`int32`, `category`, `int8`) et verification des colonnes attendues.

### 2. Fusion des tables
LEFT JOIN entre `collaborative_books_df` et `collaborative_book_metadata` sur `book_id_mapping`. Toutes les interactions sont conservees, meme sans metadonnees.

### 3. Nettoyage
Suppression des doublons, verification de la coherence des titres, creation des index continus `user_idx` et `book_idx`.

### 4. Analyse exploratoire (EDA)
Distribution des notes, top 15 des livres les plus notes, analyse du cold-start (utilisateurs et livres), heatmap de la matrice d'interactions, diagnostic du biais de notation et audit de qualite.

### 5. Detection de doublons
Recherche de livres dupliques par titre exact et par similarite semantique (SequenceMatcher > 85%).

### 6. Modele Baseline
Score de popularite ponderee inspire de la formule IMDb :

```
score(b) = (v / (v + m)) * R_b + (m / (v + m)) * R_global
```

Ou `v` est le nombre de votes du livre, `m` le percentile 60 du nombre de votes, `R_b` la note moyenne du livre et `R_global` la moyenne globale (3.923).

### 7. Content-Based (TF-IDF)
Construction d'un corpus textuel pour chaque livre (`titre * 3 + auteur * 3 + genre * 2 + description`) puis vectorisation TF-IDF (matrice 96 x 1380). Les recommandations sont basees sur la similarite cosinus entre les vecteurs.

Inclut egalement un clustering K-Means (9 clusters, 32 dimensions via SVD tronquee) pour regrouper les livres par profil.

### 8. Collaborative Filtering (KNN item-based)
Construction d'une matrice creuse 66 909 x 898, puis transposition (898 x 66 909) pour comparer les livres entre eux. NearestNeighbors avec k=20 et metrique cosinus.

### 9. Matrix Factorization (SVD)
Decomposition en valeurs singulieres tronquee avec 50 composantes, incluant les biais utilisateur et livre :

```
prediction(u,b) = U @ diag(S) @ V^T + mu + bias_user + bias_book
```

**Metriques** : RMSE = 0.9997, MAE = 0.6541, Weighted RMSE = 1.0668, variance expliquee = 20.68%.

### 10. Hybrid Scorer
Fusion ponderee des trois moteurs avec normalisation min-max :

| Moteur | Poids |
|---|---|
| Content-Based (TF-IDF) | 0.30 |
| Collaborative Filtering (KNN) | 0.40 |
| Matrix Factorization (SVD) | 0.30 |

---

## Comparaison des moteurs

| Moteur | Personnalisation | Similarite | Couverture | Cold-start | RMSE |
|---|---|---|---|---|---|
| Baseline | Faible | Indirecte | 898 livres (100%) | Bon (new users) | — |
| Content-Based | Moyenne | Excellente (cosinus) | 96 livres (10.6%) | Bon (new users) | — |
| Collaborative (KNN) | Bonne | Excellente (comportementale) | Bonne si interactions | Sensible | — |
| MF (SVD) | Tres bonne | Indirecte | 898 livres | Sensible | 0.9997 |
| Hybride | Tres bonne | Tres bonne | 898 (compromis) | Partiellement reduit | — |

**Recommandation** : le modele Hybride offre le meilleur compromis pertinence/couverture. En repli, le SVD assure la personnalisation et le Content-Based gere le cold-start.

---

## Application web

L'application `bookrecommandation.html` est un fichier HTML autonome qui permet de visualiser les resultats du systeme directement dans un navigateur, sans serveur ni installation.

### Fonctionnalites

- **Vue d'ensemble** : statistiques du dataset, distribution des notes (Chart.js), metriques SVD et presentation des 5 moteurs
- **Recommandations** : resultats reels de chaque moteur (Baseline, Content-Based, Collaborative, SVD, Hybride) avec les scores du notebook
- **Comparaison** : radar chart des performances et tableau recapitulatif
- **Explorer** : recherche parmi les 50 livres les plus votes du dataset

### Lancer l'application

Ouvrir `bookrecommandation.html` dans n'importe quel navigateur. Aucune dependance externe n'est requise (Chart.js est charge via CDN).

---

## Structure du projet

```
BookRecommandation/
├── Dataset/
│   ├── collaborative_books_df.csv
│   ├── collaborative_book_metadata.csv
│   ├── book_id_map.csv
│   ├── user_id_map.csv
│   └── book_titles.csv
├── Recommandation_de_livre_(1).ipynb
├── bookrecommandation.html
└── README.md
```

---

## Technologies utilisees

- **Python 3** : pandas, numpy, scikit-learn, scipy, matplotlib, seaborn
- **Scikit-learn** : TfidfVectorizer, NearestNeighbors, TruncatedSVD, KMeans, cosine_similarity
- **HTML / CSS / JavaScript** : application web vanilla
- **Chart.js** : visualisations (bar chart, radar chart)

---

## Installation et execution du notebook

```bash
# Cloner le projet
git clone https://github.com/<votre-username>/BookRecommandation.git
cd BookRecommandation

# Installer les dependances
pip install pandas numpy scikit-learn scipy matplotlib seaborn

# Placer les fichiers CSV dans le dossier Dataset/
# Puis ouvrir le notebook
jupyter notebook Recommandation_de_livre_(1).ipynb
```

---

## Source des donnees

Dataset : [Book Recommender System (Item-Based)](https://www.kaggle.com/datasets/thedevastator/book-recommender-system-itembased) sur Kaggle.

---

## Licence

Projet academique.
