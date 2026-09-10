# Customer Churn Prediction

![XGBoost](https://img.shields.io/badge/modèle-XGBoost-EA6C00)
![ROC-AUC 0.866](https://img.shields.io/badge/ROC--AUC-0,866-blue)
[![Démo interactive](https://img.shields.io/badge/démo-navigateur-brightgreen)](https://juleescourne.github.io/portfolio-data-analyst/#/churn)
[![Licence MIT](https://img.shields.io/badge/licence-MIT-lightgrey)](LICENSE)

Étude de cas complète sur la prédiction du départ de clients bancaires : détection
d'une fuite de données, construction de variables métier, classification XGBoost, et
arbitrage explicite du seuil de décision.

**▶ [Essayer la démo](https://juleescourne.github.io/portfolio-data-analyst/#/churn)** —
le modèle tourne dans votre navigateur, avec les contributions SHAP par prédiction.

> Projet issu de mon portfolio Data — [juleescourne.github.io/portfolio-data-analyst](https://juleescourne.github.io/portfolio-data-analyst/)

---

## Le résultat dont je suis le plus satisfait n'est pas le score

Le jeu de données contient une colonne `complain` **corrélée à 1,00 avec la cible**.

Une réclamation est enregistrée au moment du départ du client, ou après. Elle n'est
donc pas disponible à l'instant où la prédiction serait utile : le modèle
« prédirait » le passé. Conservée, elle produit un modèle à **99 % de justesse et
sans aucune valeur opérationnelle**.

Le notebook 02 l'identifie et la retire avant toute modélisation.

**Toutes les métriques ci-dessous sont donc plus basses que ce que ce jeu de données
peut afficher. C'est délibéré.**

---

## Résultats

| Métrique | Valeur | Comment la lire |
| --- | ---: | --- |
| **ROC-AUC** | **0,866** | **la seule indépendante du seuil** |
| Rappel churn | 0,90 | le seuil a été *choisi* pour l'atteindre — ce n'est pas un résultat indépendant |
| Précision churn | ≈ 0,36 | environ deux contacts sur trois seront inutiles |
| Variables retenues | 9 | sur une trentaine construites |

### Pourquoi cette précision de 36 % est un choix, pas un échec

Perdre un client coûte plus cher que contacter inutilement un client fidèle. Le
seuil est déplacé pour capter le maximum de départs, au prix de faux positifs.

Formulé côté métier : pour capter 9 clients à risque sur 10, l'équipe rétention
contacte environ **2,8 clients pour chaque départ réellement évité**. Le compromis
tient si le coût d'un contact vaut moins d'un tiers de la valeur d'un client retenu.

![Compromis précision / rappel](assets/precision_recall_threshold.png)

---

## Ce que le projet démontre

| Domaine | Éléments concrets |
| --- | --- |
| Jugement analytique | détection et exclusion d'une variable en fuite corrélée à 1,00 |
| Feature engineering | drapeaux de risque, interactions, ratios métier, scores d'engagement |
| Modélisation | XGBoost, gestion du déséquilibre, sélection de variables, arbitrage du seuil |
| Traduction métier | le seuil est justifié par un coût relatif, pas par une cible arbitraire |
| Industrialisation | export ONNX et table SHAP pour une inférence navigateur |
| Honnêteté | les biais méthodologiques sont documentés, pas dissimulés |

![Importance des variables](assets/feature_importance.png)

---

## Les biais que je n'ai pas masqués

Cette section coûte des points en apparence et en gagne en entretien.

**Le rappel de 0,90 est partiellement tautologique.** Le notebook cherche le point
de la courbe où le rappel vaut 0,9, puis rapporte 0,90. La métrique porteuse
d'information est le ROC-AUC.

**Le seuil est choisi sur l'échantillon d'évaluation.** Il faudrait un jeu de
validation dédié, puis une seule évaluation sur un test intact.

**Les variables sont conçues sur le jeu complet.** Seuils des drapeaux de risque,
sélection des interactions, quantiles : tous calculés avant découpage. Les
métriques sont donc optimistes.

**Pas de validation croisée ni de modèle de référence.** Sans régression logistique
de comparaison, un ROC-AUC de 0,866 n'a pas d'échelle.

Détail et correctifs dans
[ARCHITECTURE.md](ARCHITECTURE.md#5-biais-méthodologiques-identifiés).

---

## Démarrage rapide

```bash
git clone https://github.com/juleescourne/customer-churn-prediction.git
cd customer-churn-prediction
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Téléchargez ensuite le [jeu de données Kaggle](https://www.kaggle.com/datasets/kartiksaini18/churn-bank-customer)
vers `data/raw/Customer-Churn-Records.csv`, puis exécutez les quatre notebooks dans
l'ordre.

Détail : [INSTALLATION.md](INSTALLATION.md).

---

## Documentation

| Document | Contenu |
| --- | --- |
| [INSTALLATION.md](INSTALLATION.md) | installation, jeu de données, artefacts de démonstration, dépannage |
| [UTILISATION.md](UTILISATION.md) | rôle de chaque notebook, lecture des métriques, résultats métier |
| [ARCHITECTURE.md](ARCHITECTURE.md) | chaîne de traitement, variables, biais méthodologiques, feuille de route |

---

## Stack

`Python 3.11+` · `pandas` · `NumPy` · `scikit-learn` · `XGBoost` · `SHAP`
· `Matplotlib` · `Seaborn` · `Jupyter` · `ONNX Runtime Web`

---

## Structure

```text
notebooks/    01 nettoyage · 02 exploration · 03 features · 04 modélisation
scripts/      export_demo_artifacts.py — ONNX + table SHAP pour la démo
assets/       figures utilisées dans la documentation
data/         jeu source et intermédiaires (exclus de Git)
model/        modèle et liste de variables (exclus de Git)
```

---

## Licence

[MIT](LICENSE) — Jules Courné. Le jeu de données Kaggle n'est pas couvert par cette
licence et n'est pas redistribué.
