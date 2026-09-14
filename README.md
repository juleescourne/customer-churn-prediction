# Churn bancaire : prioriser les clients à examiner

Classification sur 10 000 clients : régression logistique et XGBoost, choix sur
validation et évaluation sur test séparé. Le modèle priorise des contacts ; il ne
mesure pas les départs évités.

## Résultats reproductibles

| Modèle | ROC-AUC test | Average precision test |
| --- | ---: | ---: |
| Régression logistique | 0.7752 | 0.4806 |
| XGBoost | 0.8686 | 0.7161 |

XGBoost est retenu sur la validation. Le seuil 0.1381 vise un rappel
≥ 85 % sur validation. Sur les 2 000 clients du test : **348 départs détectés,
59 manqués et 511 fausses alertes**, soit rappel 85.5% et précision
40.5%. Le seuil de la baseline logistique reste à 0,5 ; comparer le
classement des modèles via ROC-AUC et average precision, pas leurs rappels à ces
seuils différents.

859 alertes pour 348 départs observés ne signifient pas 348 clients retenus. Coût
du contact et efficacité d'une campagne doivent être mesurés, avec groupe témoin.

[Rapport, empreinte des données et versions](reports/evaluation.json)
· [Script exécuté](scripts/evaluate.py)

## Reproduire

Python **3.12 ou supérieur** est nécessaire pour les versions figées dans `requirements-eval.txt`.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-eval.txt
python scripts/evaluate.py --data data/raw/Customer-Churn-Records.csv
```

Télécharger le [dataset Kaggle indiqué par le projet](https://www.kaggle.com/datasets/kartiksaini18/churn-bank-customer).
L'archive utilisée contient `Churn_Modelling.csv`, à placer au chemin ci-dessus.
Son SHA-256 figure dans le rapport. Ce fichier comporte 14 colonnes et **aucune
réclamation** : il diffère du fichier enrichi exploré dans les anciens notebooks.
Les anciennes et nouvelles métriques ne mesurent pas le même protocole.
Le dataset n'est pas redistribué.

## Vérification automatisée

```bash
python -m unittest discover -s tests -v
```

La CI exécute le script sur un petit jeu synthétique : elle vérifie les contrats
d’entrée et la cohérence du protocole, sans téléchargement Kaggle. Ces tests ne
recalculent pas les scores du rapport sur les données publiques.

## Protocole

- Découpage stratifié 60/20/20, graine 42.
- Dix variables fixées à l'avance ; nom, identifiants, réclamation et satisfaction
  exclus. Dans un système réel, vérifier la disponibilité de chaque variable à
  l'instant de prédiction.
- Imputation, standardisation et encodage ajustés sur train seulement.
- Modèle et seuil choisis sur validation avant la lecture du test.

## Exploration et démonstration historiques

[Démo navigateur](https://juleescourne.github.io/portfolio-data-analyst/#/churn) :
ONNX et SHAP d'un **modèle historique distinct**, auquel les résultats ci-dessus
ne sont pas attribués. Les anciens scores (AUC 0,866, rappel proche de 90 %)
comportaient des biais de conception des variables et de sélection du seuil.
Les notebooks sont conservés comme trace pédagogique.

Source historique confirmée : [Bank Customer Churn — Radheshyam Kollipara](https://www.kaggle.com/datasets/radheshyamkollipara/bank-customer-churn), fichier `Customer-Churn-Records.csv` (10 000 lignes, 18 colonnes). Voir [le détail des deux sources](data/README.md). Les scores de référence ci-dessus restent ceux du fichier à 14 colonnes indiqué dans « Reproduire ».

L'exploration a identifié une variable de réclamation presque identique à la cible.
Cela motive son exclusion ; la corrélation seule ne prouve pas sa date d'enregistrement.

## Limites

Données déjà explorées : une partition reproductible ne remplace pas une cohorte
externe inédite. Pas de validation temporelle, de calibration évaluée ou de mesure
de rétention. Les probabilités ne sont pas des certitudes individuelles.

[Installation](INSTALLATION.md) · [Utilisation](UTILISATION.md) · [Architecture historique](ARCHITECTURE.md)

Code sous [licence MIT](LICENSE). Jules Courné.
