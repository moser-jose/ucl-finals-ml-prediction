import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix, make_scorer
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 1. Carregar dados
df = pd.read_csv('UCL_Finals_1955-2023.csv')

# Corrigir colunas duplicadas e renomear para evitar ambiguidade
#df = df.loc[:, ~df.columns.duplicated()]
# Renomear 'Attend­ance' para 'Attendance' (remover caracteres invisíveis)
#df = df.rename(columns=lambda x: x.replace('\u00ad', '').replace('­', '').replace('Attend­ance', 'Attendance'))

# Transformar 'extra_time' em binária: 1 se 'Prolongamento', 0 se 'Normal'
df['extra_time'] = df['extra_time'].apply(lambda x: 1 if str(x).strip().lower().startswith('prolong') else 0)

# 2. Selecionar features e target
features = ['Country', 'Winners', 'Runners-up', 'Venue', 'Attendance', 'Season', 'SW', 'SR', 'SW-SR']
X = df[features]
y = df['extra_time']

# 3. Pré-processamento: One-Hot Encoding para categóricas, nada para numéricas
categorical = ['Country', 'Winners', 'Runners-up', 'Venue', 'Season']
numeric = ['Attendance', 'SW', 'SR', 'SW-SR']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical),
        ('num', 'passthrough', numeric)
    ]
)

# 4. Modelos
def get_models():
    return {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(),
        'KNN - 7': KNeighborsClassifier(n_neighbors=7),
        'KNN - 5': KNeighborsClassifier(n_neighbors=5),
        'KNN - 3': KNeighborsClassifier(n_neighbors=3),
        'Random Forest': RandomForestClassifier(n_estimators=100)
    }

# 5. Avaliação com validação cruzada
def evaluate_models_cv(X, y, preprocessor, cv=10):
    models = get_models()
    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'precision': make_scorer(precision_score, zero_division=0),
        'recall': make_scorer(recall_score, zero_division=0),
        'f1': make_scorer(f1_score, zero_division=0),
        'roc_auc': 'roc_auc'
    }
    results = { 'Modelo': [], 'Acurácia': [], 'Precisão': [], 'Recall': [], 'F1-score': [], 'ROC AUC': [],
                'Acurácia_std': [], 'Precisão_std': [], 'Recall_std': [], 'F1-score_std': [], 'ROC AUC_std': [] }
    roc_curves = {}
    confusion_matrices = {}
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    for name, model in models.items():
        pipe = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
        cv_results = cross_validate(pipe, X, y, cv=skf, scoring=scoring, return_estimator=True, return_train_score=False)
        results['Modelo'].append(name)
        for metric, key in zip(['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
                               ['Acurácia', 'Precisão', 'Recall', 'F1-score', 'ROC AUC']):
            results[key].append(np.mean(cv_results[f'test_{metric}']))
            results[f'{key}_std'].append(np.std(cv_results[f'test_{metric}']))
        # ROC curve (mean)
        tprs = []
        mean_fpr = np.linspace(0, 1, 100)
        # Confusion matrix aggregation
        agg_cm = np.zeros((2, 2), dtype=int)
        for est, (_, test_idx) in zip(cv_results['estimator'], skf.split(X, y)):
            X_test_fold = X.iloc[test_idx]
            y_test_fold = y.iloc[test_idx]
            y_pred_fold = est.predict(X_test_fold)
            cm = confusion_matrix(y_test_fold, y_pred_fold, labels=[0, 1])
            agg_cm += cm
            if hasattr(est.named_steps['classifier'], 'predict_proba'):
                y_proba = est.predict_proba(X_test_fold)[:, 1]
            elif hasattr(est.named_steps['classifier'], 'decision_function'):
                y_proba = est.decision_function(X_test_fold)
            else:
                continue
            fpr, tpr, _ = roc_curve(y_test_fold, y_proba)
            tpr_interp = np.interp(mean_fpr, fpr, tpr)
            tpr_interp[0] = 0.0
            tprs.append(tpr_interp)
        if tprs:
            mean_tpr = np.mean(tprs, axis=0)
            mean_tpr[-1] = 1.0
            roc_curves[name] = (mean_fpr, mean_tpr, np.mean(cv_results['test_roc_auc']))
        confusion_matrices[name] = agg_cm
    return pd.DataFrame(results), roc_curves, confusion_matrices

# 6. Plotagem das métricas
def plot_metrics_bar(results_df, metric, filename):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(results_df['Modelo'], results_df[metric], yerr=results_df[f'{metric}_std'], capsize=5,
           color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'])
    ax.set_title(f'{metric} (média ± std)')
    ax.set_ylim(0, 1)
    ax.set_ylabel(metric)
    ax.set_xticklabels(results_df['Modelo'], rotation=15)
    plt.tight_layout()
    fig.savefig(filename)
    plt.close(fig)

# 7. Plotagem da curva ROC média
def plot_roc_curves(roc_curves, filename):
    plt.figure(figsize=(7, 6))
    for name, (fpr, tpr, auc) in roc_curves.items():
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Curva ROC Média dos Modelos (10-fold CV)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_confusion_matrix(cm, model_name, filename):
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predito')
    plt.ylabel('Real')
    plt.title(f'Matriz de Confusão Agregada - {model_name}')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# 8. Execução principal
def main():
    results_df, roc_curves, confusion_matrices = evaluate_models_cv(X, y, preprocessor, cv=10)
    metrics = ['Acurácia', 'Precisão', 'Recall', 'F1-score', 'ROC AUC']
    metric_filenames = {
        'Acurácia': 'acuracia_modelos.png',
        'Precisão': 'precisao_modelos.png',
        'Recall': 'recall_modelos.png',
        'F1-score': 'f1score_modelos.png',
        'ROC AUC': 'rocauc_modelos.png'
    }
    for metric in metrics:
        plot_metrics_bar(results_df, metric, metric_filenames[metric])
    plot_roc_curves(roc_curves, 'curva_roc.png')
    # Plotar e salvar matrizes de confusão agregadas
    for model_name, cm in confusion_matrices.items():
        model_filename = model_name.lower().replace(' ', '_')
        plot_confusion_matrix(cm, model_name, f'confusion_matrix_{model_filename}.png')
    # Imprimir métricas principais
    for i, row in results_df.iterrows():
        print(f"\nModelo: {row['Modelo']}")
        print(f"Acurácia: {row['Acurácia']:.3f} ± {row['Acurácia_std']:.3f}")
        print(f"Precisão: {row['Precisão']:.3f} ± {row['Precisão_std']:.3f}")
        print(f"Recall: {row['Recall']:.3f} ± {row['Recall_std']:.3f}")
        print(f"F1-score: {row['F1-score']:.3f} ± {row['F1-score_std']:.3f}")
        print(f"ROC AUC: {row['ROC AUC']:.3f} ± {row['ROC AUC_std']:.3f}")
    # Salvar resultados detalhados
    results_df.to_csv('resultados_validacao_cruzada.csv', index=False)

if __name__ == "__main__":
    main()