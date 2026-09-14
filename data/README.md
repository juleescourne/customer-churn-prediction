# Sources et fichiers attendus

## Évaluation de référence

Télécharger [Bank Customer Churn](https://www.kaggle.com/datasets/kartiksaini18/churn-bank-customer).
L’archive utilisée contient `Churn_Modelling.csv` : 10 000 lignes, 14 colonnes.
Le placer à `data/raw/Customer-Churn-Records.csv`, puis lancer `scripts/evaluate.py`.
Le rapport `reports/evaluation.json` conserve l’empreinte SHA-256 du fichier utilisé.

## Notebooks historiques

Source : [Bank Customer Churn — Radheshyam Kollipara](https://www.kaggle.com/datasets/radheshyamkollipara/bank-customer-churn), confirmée par l’auteur du projet.

Archive vérifiée le 14 septembre 2026 : `Customer-Churn-Records.csv`, **10 000 lignes et 18 colonnes**. Les quatre colonnes supplémentaires sont `Complain`, `Satisfaction Score`, `Card Type` et `Point Earned`.

SHA-256 du CSV historique téléchargé :
```text
2abdc051d9170540777bf35652ccb02a6ce9c9645892b078762cb729039e44d2
```

Les notebooks attendent ce fichier dans `data/raw/Customer-Churn-Records.csv`.
Conserver une copie distincte du CSV à 14 colonnes : le rapport actuel documente
l’évaluation de ce dernier, pas une réévaluation du fichier historique enrichi.
L’archive et son schéma ont été vérifiés ; les notebooks historiques n’ont pas été
réexécutés intégralement lors de cette vérification.
Les sources et les CSV intermédiaires ne sont pas versionnés.
