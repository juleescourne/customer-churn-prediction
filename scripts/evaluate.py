"""Raw-data evaluation; browser demo remains a distinct historical model."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

FEATURES = ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance',
            'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']

def evaluate(path, output):
    df = pd.read_csv(path)
    required = FEATURES + ['Exited']
    if not set(required) <= set(df): raise ValueError('Expected original bank churn columns')
    if not df.Exited.isin([0, 1]).all(): raise ValueError('Exited must be binary and complete')
    if df.Exited.nunique() != 2: raise ValueError('Both target classes are required')
    if 'CustomerId' in df and (df.CustomerId.isna().any() or df.CustomerId.duplicated().any()):
        raise ValueError('CustomerId must be unique and complete')
    for column in [c for c in FEATURES if c not in ['Geography', 'Gender']]:
        df[column] = pd.to_numeric(df[column], errors='raise')
        if np.isinf(df[column]).any(): raise ValueError(f'Infinite values in {column}')
    X, y = df[FEATURES], df.Exited
    trainval, test = train_test_split(np.arange(len(df)), test_size=.2, stratify=y, random_state=42)
    train, validation = train_test_split(trainval, test_size=.25, stratify=y.iloc[trainval], random_state=42)
    numeric = [c for c in FEATURES if c not in ['Geography', 'Gender']]
    def preprocess():
        return ColumnTransformer([
            ('numeric', make_pipeline(SimpleImputer(strategy='median'), StandardScaler()), numeric),
            ('category', make_pipeline(SimpleImputer(strategy='most_frequent'), OneHotEncoder(handle_unknown='ignore', sparse_output=False)), ['Geography', 'Gender'])])
    models = {'logistic_regression': LogisticRegression(max_iter=2000, random_state=42),
              'xgboost': XGBClassifier(n_estimators=200, max_depth=3, learning_rate=.05, subsample=.9,
                                      colsample_bytree=.9, n_jobs=2, random_state=42, eval_metric='logloss')}
    fitted, validation_ap = {}, {}
    for name, estimator in models.items():
        model = make_pipeline(preprocess(), estimator).fit(X.iloc[train], y.iloc[train])
        fitted[name] = model
        validation_ap[name] = float(average_precision_score(y.iloc[validation], model.predict_proba(X.iloc[validation])[:, 1]))
    selected = max(validation_ap, key=validation_ap.get)
    probabilities = fitted[selected].predict_proba(X.iloc[validation])[:, 1]
    thresholds = np.unique(np.r_[0., probabilities])
    threshold = max(t for t in thresholds if recall_score(y.iloc[validation], probabilities >= t) >= .85)
    scores = {}
    for name, model in fitted.items():
        probabilities = model.predict_proba(X.iloc[test])[:, 1]
        cutoff = float(threshold) if name == selected else .5
        predicted = probabilities >= cutoff
        scores[name] = dict(roc_auc=float(roc_auc_score(y.iloc[test], probabilities)),
                           average_precision=float(average_precision_score(y.iloc[test], probabilities)),
                           precision=float(precision_score(y.iloc[test], predicted, zero_division=0)),
                           recall=float(recall_score(y.iloc[test], predicted)), threshold=cutoff,
                           confusion_matrix=confusion_matrix(y.iloc[test], predicted).tolist())
    report = dict(source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), rows=len(df),
                  split=dict(train=len(train), validation=len(validation), test=len(test), seed=42),
                  features=FEATURES, ignored_columns=[c for c in df if c not in required],
                  validation_average_precision=validation_ap, selected_model=selected,
                  threshold_rule='Highest validation threshold achieving recall >= 0.85',
                  test_prevalence=float(y.iloc[test].mean()), test=scores,
                  versions=dict(sklearn=sklearn.__version__, xgboost=xgboost.__version__),
                  limitations=['Random split; no temporal validation or deployment claim.',
                               'Detected churn is not prevented churn; no campaign measured.',
                               'Browser ONNX demo is a separate historical model.',
                               'Previously explored data; this partition is not an external cohort.'])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=Path('data/raw/Customer-Churn-Records.csv'))
    parser.add_argument('--output', type=Path, default=Path('reports/evaluation.json'))
    args = parser.parse_args()
    evaluate(args.data, args.output)
