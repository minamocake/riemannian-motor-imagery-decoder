"""
Riemannian Motor Imagery Decoder - 4-class EEG classification
BNCI2014_001 dataset, Tangent Space + LogReg
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report,
                             roc_auc_score, roc_curve, auc)
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

warnings.filterwarnings("ignore")


class RawVector(BaseEstimator, TransformerMixin):
    """Flatten trials for baseline"""
    def fit(self, X, y=None): return self
    def transform(self, X): return X.reshape(X.shape[0], -1)


def load_subject(subj_id):
    dataset = BNCI2014_001()
    paradigm = MotorImagery(n_classes=4)
    X, y, meta = paradigm.get_data(dataset=dataset, subjects=[subj_id])
    le = LabelEncoder()
    return X, le.fit_transform(y), le.classes_, meta


def eval_single(subj_id=1, test_size=0.2):
    print(f"\n=== Subject {subj_id} ===")
    
    X, y, classes, _ = load_subject(subj_id)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, 
                                               random_state=42, stratify=y)
    print(f"train: {len(y_tr)}, test: {len(y_te)}")
    
    # baseline
    base = make_pipeline(RawVector(), StandardScaler(), 
                         LogisticRegression(max_iter=2000))
    grid_b = GridSearchCV(base, {'logisticregression__C': [0.1, 1, 10, 100]}, 
                          cv=3, n_jobs=-1)
    grid_b.fit(X_tr, y_tr)
    
    # riemannian
    rie = make_pipeline(Covariances(), TangentSpace(metric='riemann'),
                        LogisticRegression(max_iter=2000))
    grid_r = GridSearchCV(rie, {
        'covariances__estimator': ['oas', 'lwf', 'scm'],
        'logisticregression__C': [0.1, 1, 10, 100]
    }, cv=3, n_jobs=-1)
    grid_r.fit(X_tr, y_tr)
    
    # cv scores
    cv_b = cross_val_score(grid_b.best_estimator_, X_tr, y_tr, cv=5)
    cv_r = cross_val_score(grid_r.best_estimator_, X_tr, y_tr, cv=5)
    print(f"CV - base: {cv_b.mean():.1%}, rie: {cv_r.mean():.1%}")
    
    # test
    pred_b = grid_b.predict(X_te)
    pred_r = grid_r.predict(X_te)
    prob_r = grid_r.predict_proba(X_te)
    
    acc_b, acc_r = accuracy_score(y_te, pred_b), accuracy_score(y_te, pred_r)
    y_bin = label_binarize(y_te, classes=range(len(classes)))
    auc_m = roc_auc_score(y_bin, prob_r, multi_class='ovr', average='macro')
    
    print(f"Test - base: {acc_b:.1%}, rie: {acc_r:.1%}, AUC: {auc_m:.3f}")
    print(classification_report(y_te, pred_r, target_names=classes))
    
    return dict(classes=classes, y_te=y_te, pred=pred_r, prob=prob_r,
                cv_b=cv_b, cv_r=cv_r, acc_b=acc_b, acc_r=acc_r, 
                auc=auc_m, model=grid_r, X_te=X_te)


def eval_loso(subjs=range(1,10)):
    print("\n=== LOSO ===")
    
    dataset = BNCI2014_001()
    paradigm = MotorImagery(n_classes=4)
    X, y, meta = paradigm.get_data(dataset=dataset, subjects=list(subjs))
    
    le = LabelEncoder()
    y = le.fit_transform(y)
    subj_ids = meta['subject'].values
    
    accs = []
    for s in subjs:
        mask = subj_ids == s
        X_tr, X_te = X[~mask], X[mask]
        y_tr, y_te = y[~mask], y[mask]
        
        clf = make_pipeline(Covariances(estimator='oas'),
                           TangentSpace(metric='riemann'),
                           LogisticRegression(C=1.0, max_iter=2000))
        clf.fit(X_tr, y_tr)
        acc = accuracy_score(y_te, clf.predict(X_te))
        accs.append(acc)
        print(f"  S{s}: {acc:.1%}")
    
    print(f"Mean: {np.mean(accs):.1%} ± {np.std(accs):.1%}")
    return list(subjs), accs


def plot_all(res, save='results.png'):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    classes = res['classes']
    n_cls = len(classes)
    
    # cv comparison
    ax = axes[0,0]
    bp = ax.boxplot([res['cv_b'], res['cv_r']], patch_artist=True,
                    labels=['Baseline', 'Riemannian'])
    bp['boxes'][0].set_facecolor('lightgray')
    bp['boxes'][1].set_facecolor('#4c72b0')
    ax.axhline(0.25, color='r', ls='--', label='chance')
    ax.set_ylabel('Accuracy')
    ax.set_title('CV Comparison')
    ax.legend()
    ax.grid(axis='y', ls='--', alpha=0.5)
    
    # confusion
    ax = axes[0,1]
    cm = confusion_matrix(res['y_te'], res['pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_title(f"Confusion (acc={res['acc_r']:.1%})")
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')
    
    # roc
    ax = axes[0,2]
    y_bin = label_binarize(res['y_te'], classes=range(n_cls))
    for i, c in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_bin[:,i], res['prob'][:,i])
        ax.plot(fpr, tpr, lw=2, label=f'{c} ({auc(fpr,tpr):.2f})')
    ax.plot([0,1], [0,1], 'k--')
    ax.set_xlabel('FPR')
    ax.set_ylabel('TPR')
    ax.set_title(f'ROC (macro={res["auc"]:.2f})')
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(ls='--', alpha=0.5)
    
    # tangent pca
    ax = axes[1,0]
    est = res['model'].best_params_['covariances__estimator']
    cov = Covariances(estimator=est).fit_transform(res['X_te'])
    tang = TangentSpace(metric='riemann').fit_transform(cov)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(tang)
    sc = ax.scatter(X_pca[:,0], X_pca[:,1], c=res['y_te'], cmap='viridis', 
                    alpha=0.7, edgecolor='k', s=50)
    ax.set_title(f'Tangent PCA ({sum(pca.explained_variance_ratio_):.0%})')
    ax.set_xlabel(f'PC1')
    ax.set_ylabel(f'PC2')
    ax.legend(*sc.legend_elements(), title='Class')
    
    # gridsearch
    ax = axes[1,1]
    cv_res = res['model'].cv_results_
    Cs = [0.1, 1, 10, 100]
    for est in ['oas', 'lwf', 'scm']:
        scores = []
        for c in Cs:
            idx = [i for i,p in enumerate(cv_res['params']) 
                   if p['covariances__estimator']==est and p['logisticregression__C']==c][0]
            scores.append(cv_res['mean_test_score'][idx])
        ax.plot(range(len(Cs)), scores, 'o-', label=est, lw=2)
    ax.set_xticks(range(len(Cs)))
    ax.set_xticklabels(Cs)
    ax.set_xlabel('C')
    ax.set_ylabel('CV Acc')
    ax.set_title('Hyperparam Search')
    ax.legend()
    ax.grid(ls='--', alpha=0.5)
    
    # per-class
    ax = axes[1,2]
    rep = classification_report(res['y_te'], res['pred'], 
                                target_names=classes, output_dict=True)
    x = np.arange(n_cls)
    w = 0.25
    for i, m in enumerate(['precision', 'recall', 'f1-score']):
        vals = [rep[c][m] for c in classes]
        ax.bar(x + i*w, vals, w, label=m)
    ax.set_xticks(x + w)
    ax.set_xticklabels(classes, rotation=30)
    ax.set_ylabel('Score')
    ax.set_title('Per-Class')
    ax.legend()
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(save, dpi=150)
    plt.show()


def plot_loso(subjs, accs, save='loso.png'):
    plt.figure(figsize=(10, 5))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(subjs)))
    bars = plt.bar(subjs, accs, color=colors, edgecolor='k')
    
    mean_acc = np.mean(accs)
    plt.axhline(mean_acc, color='r', ls='--', lw=2, label=f'mean={mean_acc:.1%}')
    plt.axhline(0.25, color='gray', ls=':', label='chance')
    
    for b, a in zip(bars, accs):
        plt.text(b.get_x() + b.get_width()/2, a + 0.01, f'{a:.0%}', 
                 ha='center', fontsize=9)
    
    plt.xlabel('Subject')
    plt.ylabel('Accuracy')
    plt.title('LOSO Cross-Validation')
    plt.xticks(subjs, [f'S{s}' for s in subjs])
    plt.legend()
    plt.grid(axis='y', ls='--', alpha=0.5)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(save, dpi=150)
    plt.show()


if __name__ == "__main__":
    res = eval_single(subj_id=1)
    plot_all(res)
    
    subjs, accs = eval_loso()
    plot_loso(subjs, accs)
    
    print(f"\n--- Summary ---")
    print(f"S1: {res['acc_r']:.1%} (AUC={res['auc']:.3f})")
    print(f"LOSO: {np.mean(accs):.1%}")
