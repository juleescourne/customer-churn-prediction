> **Évaluation de référence :** `scripts/evaluate.py`, exécuté sur les données brutes.
> Voir le [rapport](reports/evaluation.json) et le [README](README.md). Les notebooks
> et résultats décrits ci-dessous concernent l’exploration historique.

# Installation

Temps nécessaire : **5 minutes**, hors téléchargement du jeu de données Kaggle.

## Prérequis

| Outil | Version | Vérifier |
| --- | --- | --- |
| Python | 3.12 ou supérieur (évaluation de référence) | `python --version` |
| Compte Kaggle | gratuit | pour télécharger le jeu de données |

---

## 1. Récupérer le projet

```bash
git clone https://github.com/juleescourne/customer-churn-prediction.git
cd customer-churn-prediction
```

## 2. Environnement virtuel

```bash
python -m venv .venv
source .venv/bin/activate          # .\.venv\Scripts\Activate.ps1 sous Windows
pip install -r requirements.txt
```

## 3. Choisir le bon fichier source

Pour **l’évaluation de référence**, installer `requirements-eval.txt`, télécharger
[Bank Customer Churn](https://www.kaggle.com/datasets/kartiksaini18/churn-bank-customer),
puis placer `Churn_Modelling.csv` à `data/raw/Customer-Churn-Records.csv`.
Ce fichier compte 10 000 lignes et **14 colonnes**. Exécuter :

```bash
python scripts/evaluate.py --data data/raw/Customer-Churn-Records.csv
```

Les notebooks historiques ci-dessous attendent un **fichier enrichi à 18 colonnes**.
Sa provenance exacte reste à confirmer : le téléchargement ci-dessus ne permet
pas de rejouer ces notebooks à l’identique. Ne pas fabriquer les colonnes manquantes.

## 4. Exploration historique — uniquement avec le fichier enrichi

```bash
jupyter notebook
```

Puis exécutez-les **dans l'ordre** — chacun produit l'entrée du suivant :

```text
notebooks/01_data_cleaning.ipynb
notebooks/02_exploratory_analysis.ipynb
notebooks/03_feature_engineering.ipynb
notebooks/04_modeling.ipynb
```

Les fichiers intermédiaires sont écrits dans `data/preprocessed/`, le modèle final
dans `model/`. Les deux répertoires sont exclus de Git.

> Utilisez un **noyau unique** pour les quatre notebooks. Un mélange de noyaux
> produit des dépendances incohérentes et des compteurs d'exécution désordonnés.

---

## Reconstruire les artefacts de la démonstration

La démonstration du portfolio a besoin de deux fichiers, produits après le
notebook 04 :

```bash
pip install shap onnxmltools onnxconverter-common
python scripts/export_demo_artifacts.py
```

| Sortie | Rôle |
| --- | --- |
| `model/xgb_churn_model.onnx` | modèle pour `onnxruntime-web` |
| `model/shap_lookup.json` | valeurs SHAP sur les 21 216 scénarios de la grille |

Ces deux fichiers sont des **sorties de build** : ils ne sont pas versionnés ici,
mais publiés sur la branche `assets` du dépôt du portfolio.

Option utile :

```bash
python scripts/export_demo_artifacts.py --skip-onnx     # reconstruire seulement SHAP
```

---

## Vérifier l'installation

Après le notebook 04, les fichiers suivants doivent exister :

```text
data/preprocessed/03_churn_records.csv
model/xgb_churn_model.pkl
model/feature_names.json
```

---

## Problèmes courants

**`FileNotFoundError: data/raw/Customer-Churn-Records.csv`**
Le fichier Kaggle n'est pas au bon endroit, ou porte un autre nom. Le chemin doit
correspondre exactement.

**`FileNotFoundError: data/preprocessed/02_churn_records.csv`**
Vous avez sauté un notebook. Ils doivent être exécutés dans l'ordre.

**`XGBoostError: use_label_encoder`**
Paramètre supprimé de XGBoost 2.x. Retirez-le de l'appel concerné : il n'a plus
d'effet.

**`ModuleNotFoundError: No module named 'shap'`**
SHAP n'est pas dans `requirements.txt` : il ne sert qu'au script d'export.
Installez-le à part (voir plus haut).

**Le script d'export échoue sur `model/xgb_churn_model.pkl`**
Le notebook 04 n'a pas été exécuté jusqu'au bout — c'est lui qui écrit le modèle.
