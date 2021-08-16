import os
import mne
import numpy as np
import sklearn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold
import pandas as pd
from matplotlib import pyplot as plt

import p3k
from p3k.params import SpellerInfo
from p3k.classification.utils import reshape_mne_raw_for_lda

def extract_target_from_trial(epochs: mne.Epochs, trial_nb: int, stimulus_code_begin: int):
    #epoch_trial = epochs[epochs.metadata['Trial_nb'] == trial_nb]
    assert (epochs.metadata['Trial_nb'] == trial_nb).size > 0, 'Trial number not found in metadata'
    idx_targets = np.where(np.logical_and(epochs.metadata['Trial_nb'] == trial_nb,
                                          epochs.metadata['is_target']))[0]
    # extract target or target pair
    target_pair = np.sort(np.unique(epochs.metadata.loc[idx_targets]['stim'])).astype(np.uint8)
    # remove the stimulus pardding to give it a meaning
    return target_pair - stimulus_code_begin


def predict_cumulative(clf, X, step):
    """
    Repeats a function on the first axis on the 2d data"""
    array_out = [clf.predict(X[0:cursor_seq, :]) for cursor_seq in np.arange(0, X.shape[0], step) + step]
    return array_out


def predict_proba_cumulative(clf, X, step):
    """
    Repeats a function on the first axis on the 2d data"""
    array_out = [clf.predict_proba(X[0:cursor_seq, :]) for cursor_seq in np.arange(0, X.shape[0], step) + step]
    return array_out


def score_cumulative(clf, X, y, step):
    """
    Repeats a function on the first axis on the 2d data"""
    array_out = [clf.score(X[0:cursor_seq, :], y[0:cursor_seq]) for cursor_seq in np.arange(0, X.shape[0], step) + step]
    return array_out


def auc_cumulative(y, y_preds_cumulative, step):
    array_out = []
    for i in range(0, y.shape[0] // step):
        y_pred = y_preds_cumulative[i]
        y_gt = y[0:step * (i + 1)]
        roc = sklearn.metrics.roc_auc_score(y_gt, y_pred)
        array_out.append(roc)

    #one liner not working TT
    #array_out = [roc_auc_score(y[0:cursor_seq], y_preds_cumulative[(cursor_seq//step)-1]) for cursor_seq in np.arange(0,y.shape[0], step) + step]
    return array_out


# Transform into a function

def stim_from_predict_cumulative(predicted, stim_labels, nb_stim_in_sequence=None):
    """
    takes the output of predict_proba and returns the predicted target.
    :param predicted:
    :param stim_labels: stimulus labels from the epochs
    :param len_repetition: number of stimuli in a sequence
    :return: preicted target stimulus
    :note: if no target was found by LDA, it will use best candidates based
    """
    # retrieve the target probability index
    max_proba_class = np.argmax(predicted, axis=1)

    # In case LDA returns no results
    if np.max(max_proba_class) == 0:
        #print('(note:LDA detected no target when classifying n={} epochs, using best target candidate instead)'.format(predicted.shape[0]))
        #max_proba_class[[np.argmax(predicted[:,1])]] = 1
        print("- LDA detected no target in an iteration of length {}."
              " Marking best candidate as target every {} prediction".format(predicted.shape[0], nb_stim_in_sequence))
        for step_seq in range(0, max_proba_class.size, nb_stim_in_sequence):
            idx_seq_step = list(range(step_seq, step_seq + nb_stim_in_sequence))
            max_proba_class[[idx_seq_step[0] + np.argmax(predicted[idx_seq_step, 1])]] = 1

    # extract probabilities
    max_proba = np.array([predicted[idx, max_proba_class[idx]] for idx in list(range(predicted.shape[0]))])
    # target_indices
    target_epochs_idx = np.where(max_proba_class)
    # extract target stimuli from the list of stimuli and put them in a table with their respective probability
    predicted_target_stims = stim_labels[target_epochs_idx].astype(np.uint8)
    pred_targets_table = np.vstack((predicted_target_stims, max_proba[target_epochs_idx]))
    # sum up target probability in this table stimulis-wise (thus dealing with situations with draws)
    potential_targets = np.unique(pred_targets_table[0, :])
    sum_proba_targets = [np.sum(pred_targets_table[1, np.where(pred_targets_table[0, :] == target_candidate)]) for
                         target_candidate in potential_targets]
    pred_targets_table_reduced = np.vstack((potential_targets, sum_proba_targets))

    # return the stimulation with the highest average probability
    predicted_stim = pred_targets_table_reduced[0, np.argmax(pred_targets_table_reduced[1, :])].astype(np.uint8)

    return predicted_stim

def _validate_kfold(epochs: mne.Epochs, nb_k_splits: int) -> bool:
    list_trials = np.unique(epochs.metadata["Trial_nb"])

    print("nb_trials:{}, nb_folds:{}".format(list_trials.size, nb_k_splits))
    assert list_trials.size // nb_k_splits == list_trials.size / nb_k_splits, \
        f'number of splits {nb_k_splits} must be a multiple of the number of trials {list_trials.size}'
    return True


def run_p300_LDA_analysis(epochs: mne.epochs,
                          nb_k_fold: int,
                          speller_info: SpellerInfo):

    clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')

    kf = KFold(n_splits=nb_k_fold)

    # make a table to store the scores and accuracies
    score_table = pd.DataFrame({'fold': pd.Series([], dtype='int'),
                                'fold_trial': pd.Series([], dtype='int'),
                                'n_seq': pd.Series([], dtype='int'),
                                'score': pd.Series([], dtype='float'),
                                'AUC': pd.Series([], dtype='float'),
                                'row_true': pd.Series([], dtype='str'),
                                'row_pred': pd.Series([], dtype='str'),
                                'col_true': pd.Series([], dtype='str'),
                                'col_pred': pd.Series([], dtype='str'),
                                'correct': pd.Series([], dtype='int')})

    list_trials = np.unique(epochs.metadata["Trial_nb"])

    fold_counter = 0
    for train_index, test_index in kf.split(list_trials):
        #print(f"processing fold {fold_counter}")
        a = test_index

        # define the training sample
        X_train = epochs[epochs.metadata['Trial_nb'].isin(list_trials[train_index])]._data
        y_train = epochs[epochs.metadata['Trial_nb'].isin(list_trials[train_index])].metadata[
            'is_target'].astype(np.uint8)
        y_train_stim = epochs[epochs.metadata['Trial_nb'].isin(list_trials[train_index])].metadata[
            'stim']

        # reshape data to enter LDA
        X_train = reshape_mne_raw_for_lda(X_train)

        # train the LDA classifier
        clf.fit(X_train, y_train)


        # split the tests samples for each individual trial enable P300 speller target prediction and accuracy
        nb_test_trials_in_split = len(list_trials)
        for i_trial in list(range(len(test_index))):
            trial_nb = test_index[i_trial] + 1

            true_target_cols = None
            true_target_rows = None

            # both row and columns for generic classification
            epoch_test = epochs[epochs.metadata['Trial_nb'] == trial_nb]
            X_test = epoch_test._data
            y_test = epoch_test[epoch_test.metadata['Trial_nb'] == trial_nb].metadata['is_target'].astype(np.uint8)

            # separate row and columns target detection
            epoch_rows = epoch_test[np.where(epoch_test.metadata['is_row'])[0]]
            epoch_cols = epoch_test[np.where(epoch_test.metadata['is_col'])[0]]

            skip_rows = False
            if np.where(epoch_rows.metadata['is_target'] == 1)[0].size == 0:
                print('Skipping epochs with no target rows')
                skip_rows = True
            else:
                X_test_rows = epoch_rows[epoch_rows.metadata['Trial_nb'] == trial_nb]._data
                y_test_rows = epoch_rows[epoch_rows.metadata['Trial_nb'] == trial_nb].metadata['is_target'].astype(np.uint8)
                y_test_rows_stim = epoch_rows[epoch_rows.metadata['Trial_nb'] == trial_nb].metadata['stim']
                X_test_rows = reshape_mne_raw_for_lda(X_test_rows)  # reshape for LDA

            skip_cols = False
            if np.where(epoch_cols.metadata['is_target'] == 1)[0].size == 0:
                skip_cols = True
                print('Skipping epochs with no target columns')
            else:
                X_test_cols = epoch_cols[epoch_cols.metadata['Trial_nb'] == trial_nb]._data
                y_test_cols = epoch_cols[epoch_cols.metadata['Trial_nb'] == trial_nb].metadata['is_target'].astype(np.uint8)
                y_test_cols_stim = epoch_cols[epoch_cols.metadata['Trial_nb'] == trial_nb].metadata['stim']
                X_test_cols = reshape_mne_raw_for_lda(X_test_cols)  # reshape

            # reshape data to enter LDA
            X_test = reshape_mne_raw_for_lda(X_test)

            # predict the targets for rows, (1 to N sequences cumulative X information provided)
            y_pred = clf.predict(X_test)

            # update this when dealing with rows and columns to X_test_rows and X_test_cols
            step = X_test.shape[0] // speller_info.nb_seq

            # score
            score_cum = score_cumulative(clf, X_test, y_test, step)
            # probas calculated for prediction (aggregation will lead to cumulative prediction)
            #proba_cum = proba_cumulative(clf, X_test, y_test, step)
            y_test_pred_cum = predict_cumulative(clf, X_test, step)
            # area under the curve
            auc_cum = auc_cumulative(y_test, y_test_pred_cum, step)

            if not skip_rows:
                # prediction stim_rows
                step_rows = X_test_rows.shape[0] // speller_info.nb_seq
                proba_cum_rows = predict_proba_cumulative(clf, X_test_rows, step_rows)
                pred_stim_rows_cum = np.array([stim_from_predict_cumulative(predicted=proba_cum_rows[i],
                                                                            stim_labels=epoch_rows.metadata[
                                                                                'stim'].to_numpy(),
                                                                            nb_stim_in_sequence=step_rows)
                                               for i in list(range(len(proba_cum_rows)))])
                true_target_rows = np.unique(epoch_rows.metadata.iloc[
                                                 np.where(epoch_rows.metadata['is_target'] == 1)]['stim'])[0]  # does not handle k fold with several trials
                rows_successful = pred_stim_rows_cum == true_target_rows

            if not skip_cols:
                # columns
                step_cols = X_test_cols.shape[0] // speller_info.nb_seq
                proba_cum_cols = predict_proba_cumulative(clf, X_test_cols, step_cols)

                # @todo make these functions readable by human
                pred_stim_cols_cum = np.array([stim_from_predict_cumulative(predicted=proba_cum_cols[i],
                                                                            stim_labels=epoch_cols.metadata[
                                                                                'stim'].to_numpy(),
                                                                            nb_stim_in_sequence=step_cols)

                                               for i in list(range(len(proba_cum_cols)))])

                true_target_cols = np.unique(epoch_cols.metadata.iloc[
                                                 np.where(epoch_cols.metadata['is_target'] == 1)[0]]['stim'])[0]  # does not handle k fold with several trials
                cols_successful = pred_stim_cols_cum == true_target_cols

            if not skip_rows and not skip_cols:
                successful_pred_cum = np.logical_and(rows_successful, cols_successful).astype(np.uint8)
            elif skip_rows and skip_cols:
                print('Errors, cannot classify if both rows and columns are skipped')
                raise Exception
            elif skip_rows:
                successful_pred_cum = cols_successful
            elif skip_cols:
                successful_pred_cum = rows_successful


            # Associate predicted targets to stimuli
            for i in range(len(score_cum)):

                # deal with undefined columns or rows
                if true_target_cols is None:
                    col_true = -1
                    col_pred = -1
                else:
                    col_true = true_target_cols
                    col_pred = pred_stim_cols_cum[i]

                if true_target_rows is None:
                    row_true = -1
                    row_pred = -1
                else:
                    row_true = true_target_rows
                    row_pred = pred_stim_rows_cum[i]

                # write line to score table
                line = dict(
                    zip(score_table.columns, [fold_counter + 1, i_trial + 1, i + 1,  # fold, fold_trial, nb of sequences
                                              score_cum[i],
                                              auc_cum[i], row_true, row_pred,
                                              col_true, col_pred, successful_pred_cum[i]]))

                score_table = score_table.append(line, ignore_index=True)

        print('fold {} partial score: {}, AUC={}'.format(fold_counter, np.round(score_cum[-1], decimals=3),
                                                         np.round(auc_cum[-1], decimals=3)))

        #print('----row score: {}, AUC={}'.format(np.round(kscore_rows, decimals=3), np.round(auc_rows, decimals=3)))
        #print('----col score: {}, AUC={}'.format(np.round(kscore_cols, decimals=3), np.round(auc_cols, decimals=3)))
        fold_counter += 1

    # clear the types or the score table
    score_table[["row_true", "row_pred", "col_true", "col_pred"]] = score_table[["row_true", "row_pred", "col_true",
                                                                                 "col_pred"]] - 100
    score_table = score_table.convert_dtypes()

    return score_table


def plot_cum_score_table(table: pd.DataFrame,
                         nb_cross_fold: int,
                         show=True):
    df_seq = table.groupby(['n_seq']).mean()
    df_seq = df_seq.rename(columns={"correct": "Accuracy", "score": "epoch_score", "AUC": "epoch_AUC"})
    df_seq[['Accuracy', 'epoch_score', 'epoch_AUC']]

    ax = df_seq.plot(y='Accuracy')
    plt.ylim(0, 1.01)
    plt.suptitle('Cross-fold offline accuracy (n={})'.format(nb_cross_fold))
    plt.xlabel('Number of sequences')
    plt.ylabel('Accuracy')
    plt.legend(['P300 target prediction (row x col)'])
    # export the figure
    if show:
        plt.show()
    return ax.figure