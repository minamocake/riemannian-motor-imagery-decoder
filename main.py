import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve, auc
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize, FunctionTransformer
from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

# Config
SUBJECT_ID = 1
TEST_SIZE = 0.2
CV_FOLDS = 5
C_GRID = [0.1, 1, 10, 100]


def get_baseline_pipeline():
    """Euclidean baseline: flatten -> scale -> LR"""
    return make_pipeline(
        FunctionTransformer(lambda x: x.reshape(x.shape[0], -1)),
        StandardScaler(),
        LogisticRegression(max_iter=2000)
    )


def get_riemann_pipeline():
    """Riemannian: cov -> tangent space -> LR
    OAS: shrinkage estimator, stable for high-dim EEG with limited samples
    """
    return make_pipeline(
        Covariances(estimator='oas'),
        TangentSpace(metric='riemann'),
        LogisticRegression(max_iter=2000)
    )


def load_data(subj):
    dataset = BNCI2014_001()
    paradigm = MotorImagery(n_classes=4)
    X, y, meta = paradigm.get_data(dataset=dataset, subjects=[subj])
    le = LabelEncoder()
    return X, le.fit_transform(y), le.classes_, meta


def run_single(subj):
    print(f"\n--- Subject {subj} ---")
    
    X, y, classes, _ = load_data(subj)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=42, stratify=y
    )
    
    # baseline
    grid_b = GridSearchCV(get_baseline_pipeline(), 
                          {'logisticregression__C': C_GRID}, cv=CV_FOLDS, n_jobs=-1)
    grid_b.fit(X_tr, y_tr)
    
    # riemannian
    grid_r = GridSearchCV(get_riemann_pipeline(), 
                          {'logisticregression__C': C_GRID}, cv=CV_FOLDS, n_jobs=-1)
    grid_r.fit(X_tr, y_tr)
    
    # test
    pred_r = grid_r.predict(X_te)
    prob_r = grid_r.predict_proba(X_te)
    acc_b = accuracy_score(y_te, grid_b.predict(X_te))
    acc_r = accuracy_score(y_te, pred_r)
    
    y_bin = label_binarize(y_te, classes=range(len(classes)))
    auc_score = roc_auc_score(y_bin, prob_r, multi_class='ovr', average='macro')
    
    print(f"Test: base={acc_b:.1%}, rie={acc_r:.1%}, AUC={auc_score:.3f}")
    
    return {
        'classes': classes, 'y_te': y_te, 'pred': pred_r, 'prob': prob_r,
        'acc_b': acc_b, 'acc_r': acc_r, 'auc': auc_score
    }


def run_loso(subjs=range(1, 10)):
    print("\n--- LOSO ---")
    
    dataset = BNCI2014_001()
    paradigm = MotorImagery(n_classes=4)
    X, y, meta = paradigm.get_data(dataset=dataset, subjects=list(subjs))
    
    le = LabelEncoder()
    y = le.fit_transform(y)
    sids = meta['subject'].values
    
    accs = []
    for s in subjs:
        mask = sids == s
        clf = get_riemann_pipeline()
        clf.set_params(logisticregression__C=1.0)
        clf.fit(X[~mask], y[~mask])
        acc = accuracy_score(y[mask], clf.predict(X[mask]))
        accs.append(acc)
        print(f"  S{s}: {acc:.1%}")
    
    print(f"Mean: {np.mean(accs):.1%} +/- {np.std(accs):.1%}")
    return list(subjs), accs


def plot_results(res, subj):
    classes = res['classes']
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 1. Bar plot: Baseline vs Riemannian
    ax = axes[0]
    methods = ['Baseline', 'Riemannian']
    accs = [res['acc_b'], res['acc_r']]
    bars = ax.bar(methods, accs, color=['lightgray', '#4c72b0'], edgecolor='k')
    ax.axhline(0.25, c='r', ls='--', label='Chance')
    ax.set_ylim(0, 1)
    ax.set_ylabel('Accuracy')
    ax.set_title(f'Method Comparison (S{subj})')
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, acc + 0.02, f'{acc:.1%}', ha='center')
    
    # 2. Confusion matrix
    ax = axes[1]
    sns.heatmap(confusion_matrix(res['y_te'], res['pred']), annot=True, fmt='d',
                xticklabels=classes, yticklabels=classes, ax=ax, cmap='Blues')
    ax.set_title('Confusion Matrix')
    ax.set_ylabel('True')
    ax.set_xlabel('Predicted')
    
    # 3. ROC
    ax = axes[2]
    y_bin = label_binarize(res['y_te'], classes=range(len(classes)))
    for i, c in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], res['prob'][:, i])
        ax.plot(fpr, tpr, lw=2, label=f'{c} ({auc(fpr, tpr):.2f})')
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel('FPR')
    ax.set_ylabel('TPR')
    ax.legend(fontsize=8)
    ax.set_title(f'ROC (AUC={res["auc"]:.2f})')
    
    plt.tight_layout()
    plt.savefig('results.png', dpi=150)
    plt.show()


def plot_loso(subjs, accs):
    mean_acc = np.mean(accs)
    
    plt.figure(figsize=(10, 4))
    plt.bar(subjs, accs, color='#4c72b0', edgecolor='k')
    plt.axhline(mean_acc, c='r', ls='--', label=f'Mean: {mean_acc:.1%}')
    plt.axhline(0.25, c='gray', ls=':', label='Chance')
    
    for s, a in zip(subjs, accs):
        plt.text(s, a + 0.01, f'{a:.0%}', ha='center', fontsize=9)
    
    plt.xlabel('Subject')
    plt.ylabel('Accuracy')
    plt.title('LOSO Cross-Validation')
    plt.xticks(subjs, [f'S{s}' for s in subjs])
    plt.legend()
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig('loso.png', dpi=150)
    plt.show()


if __name__ == "__main__":
    res = run_single(SUBJECT_ID)
    plot_results(res, SUBJECT_ID)
    
    subjs, accs = run_loso()
    plot_loso(subjs, accs)
    
    print(f"\nS{SUBJECT_ID}: {res['acc_r']:.1%} (AUC={res['auc']:.3f})")
    print(f"LOSO: {np.mean(accs):.1%}")