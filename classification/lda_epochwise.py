from pandas import pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import model_selection, metrics
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, plot_precision_recall_curve

import seaborn as sns

from p3k.classification.utils import reshape_mne_raw_for_lda


def run_single_epoch_LDA_analysis(X_data: np.ndarray,
                                  y_true_labels: np.ndarray,
                                  nb_k_fold: int,
                                  display_confidence_matrix: bool = True,
                                  display_precision_recall: bool = True):
    # Reshaping data
    X = reshape_mne_raw_for_lda(X_data)
    y = y_true_labels
    # Separating train and test
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3)

    clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    kf = KFold(n_splits=nb_k_fold)
    kf.get_n_splits(X)

    list_score = []
    list_auc = []

    for train_index, test_index in kf.split(X):
        X_train_kf, X_test_kf = X[train_index], X[test_index]
        y_train_kf, y_test_kf = y[train_index], y[test_index]
        clf.fit(X_train_kf, y_train_kf)
        kscore = clf.score(X_test_kf, y_test_kf)
        y_pred_kf = clf.predict(X_test_kf)
        k_auc = roc_auc_score(y_test_kf, y_pred_kf)
        print('fold score: {}, AUC={}'.format(np.round(kscore, decimals=3), np.round(k_auc, decimals=3)))
        list_score.append(kscore)
        list_auc.append(k_auc)

    print('Average score {}-Fold = {}, AUC={}'.format(kf.get_n_splits(X), np.round(np.mean(list_score), decimals=2),
                                                      np.round(np.mean(list_auc), decimals=2)))

    # using the training/validate samples,
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    y_pred = clf.predict(X_test)
    auc = roc_auc_score(y_test, y_pred)

    print('Score training-validation {}, AUC={}'.format(np.round(score, decimals=2), np.round(auc, decimals=2)))

    # compare the count of targets to the amount of stimuli
    target_to_nontarget_ratio = np.sum(y) / y.shape[0]
    if target_to_nontarget_ratio != .5:
        print(f'Score is only valid if classes are balanced target_ratio={target_to_nontarget_ratio:0.2f}\n'
              f' please refer to AUC instead')

    if display_confidence_matrix:
        df_confidence, fig_confidence = conf_matrix(y=y_test, pred=y_pred)

    y_score = clf.decision_function(X_test)
    if display_precision_recall:
        fig_roc = plot_precision_recall_curve(classifier=clf, X=X_test, y_gt=y_test)
    else:
        fig_roc = None

    return fig_confidence, fig_roc

def conf_matrix(y: np.ndarray, pred: np.ndarray):
    cmat = metrics.confusion_matrix(y, pred)
    cmat_norm = metrics.confusion_matrix(y, pred,
                                         normalize='true')
    ((tn, fp), (fn, tp)) = cmat
    ((tnr, fpr), (fnr, tpr)) = cmat_norm

    fig = plt.figure()
    labels = ['Non-Target', 'Target']
    sns.heatmap(cmat, xticklabels=labels, yticklabels=labels, annot=True, fmt="d")
    plt.title('Confusion matrix')
    plt.xlabel('Ground Truth')
    plt.ylabel('Predicted')

    # alternative using sklearn plots
    # plt.figure()
    # from sklearn.metrics import ConfusionMatrixDisplay
    # cm_display = ConfusionMatrixDisplay(cmat).plot()

    return pd.DataFrame([[f'TN = {tn} (TNR = {tnr:1.2%})',
                          f'FP = {fp} (FPR = {fpr:1.2%})'],
                         [f'FN = {fn} (FNR = {fnr:1.2%})',
                          f'TP = {tp} (TPR = {tpr:1.2%})']],
                        index=['True 0(Non-Target)', 'True 1(Target)'],
                        columns=['Pred 0(Non-Target)',
                                 'Pred 1(Target)']), fig


