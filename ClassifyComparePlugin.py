from sklearn.metrics import matthews_corrcoef, f1_score, balanced_accuracy_score, auc, precision_score, recall_score
import random
import numpy as np
from sklearn.metrics import precision_recall_curve, auc, average_precision_score

# def chunk(xs, n):
#     """
#     Randomly split a list into n chunks
#     """
#     ys = list(xs)
#     random.shuffle(ys)
#     ylen = len(ys)
#     size = int(ylen / n)
#     chunks = [ys[0+size*i : size*(i+1)] for i in xrange(n)]
#     leftover = ylen - size*n
#     edge = size*n
#     for i in xrange(leftover):
#             chunks[i%n].append(ys[edge+i])
#     return chunks

import PyIO
import PyPluMA
import pandas

import pickle
def find_threshold_one_fold(df, score_name, label_name):
    precision, recall, thresholds = precision_recall_curve(df[label_name], df[score_name])
    max_matthews = 0
    optimal_threshold = 0
    labels = df[label_name]
    for thr in thresholds:
        pred_labels = df[score_name].apply(lambda x: int(x>thr))
        matthews = matthews_corrcoef(labels, pred_labels)
        if matthews>max_matthews:
            max_matthews = matthews
            optimal_threshold = thr
    return optimal_threshold, max_matthews

def find_optimal_threshold(df, score_name, label_name, reverse_sign=True):
    """
    df - dataframe with binding scores
    score_name - column name with binding scores
    label name - column name with true labels
    if reverse_sign, than we assume that lower values represent the better score
    ----
    return: optimal threshold
    """
    df = df.copy()
    if reverse_sign:
        df[score_name] = - df[score_name]

    all_thresholds = []
    all_matthews = []
    shuffled = df.sample(frac=1, random_state=15)
    all_chunks = np.array_split(shuffled, 10)
    labels = df[label_name]
    for cv_df in all_chunks:
        cv_optimal, cv_matthews = find_threshold_one_fold(cv_df, score_name, label_name)
        all_thresholds.append(cv_optimal)
        all_matthews.append(cv_matthews)

    optimal_threshold = np.mean(all_thresholds)
    pred_labels = df[score_name].apply(lambda x: int(x>optimal_threshold))
    balanced_accuracy = balanced_accuracy_score(labels, pred_labels)
    f1 = f1_score(labels, pred_labels)
    precision = precision_score(labels, pred_labels)
    recall = recall_score(labels, pred_labels)

    if reverse_sign:
        all_thresholds = [-x for x in all_thresholds]
        optimal_threshold = np.mean(all_thresholds)

    print(f"Optimal threshold: {optimal_threshold}; BA={balanced_accuracy}, F1={f1}, Precision={precision}; Recall={recall}")
    return all_thresholds, all_matthews, [balanced_accuracy, f1, precision, recall]


class ClassifyComparePlugin:
    def input(self, inputfile):
       self.parameters = PyIO.readParameters(inputfile)
    def run(self):
        pass
    def output(self, outputfile):
        # At the moment this takes four files: Three datasets, and a set of other
        # Future will be to make that part flexible
        all_results = []
        all_thresholds = []
        all_matthews = []
        if ("file1" in self.parameters):
           scores_df = pandas.read_csv(PyPluMA.prefix()+"/"+self.parameters["file1"])
           label1 = self.parameters["label1"]
           thresholds, matthews, metrics = find_optimal_threshold(scores_df, 'score', 'label')
           all_results.append([label1]+metrics)
           all_thresholds.append([label1]+thresholds)
           all_matthews.append([label1]+matthews)
        if ("file2" in self.parameters):
           SCORES_MASIF=PyPluMA.prefix()+"/"+self.parameters["file2"]#"data/masif_test/MaSIF-Search_scores.csv"
           scores_masif = pandas.read_csv(SCORES_MASIF)
           label2 = self.parameters["label2"]
           thresholds, matthews, metrics = find_optimal_threshold(scores_masif, 'score', 'label')
           all_results.append([label2]+metrics)
           all_thresholds.append([label2]+thresholds)
           all_matthews.append([label2]+matthews)
        if ("file3" in self.parameters):
           dMASIF_SCORES=PyPluMA.prefix()+"/"+self.parameters["file3"]#"data/masif_test/dmasif_out.csv"
           scores_dmasif = pandas.read_csv(dMASIF_SCORES)
           label3 = self.parameters["label3"]
           thresholds, matthews, metrics = find_optimal_threshold(scores_dmasif, 'avg_score', 'label', reverse_sign=False)
           all_results.append([label3]+metrics)
           all_thresholds.append([label3]+thresholds)
           all_matthews.append([label3]+matthews)


        OTHER_SCORES=PyPluMA.prefix()+"/"+self.parameters["other"]#"data/masif_test/Other_tools_SCORES.csv"

        #scores_df = pandas.read_csv("PIsToN_scores.csv")
        other_labels = PyIO.readSequential(PyPluMA.prefix()+"/"+self.parameters["other_labels"])
        if (OTHER_SCORES.endswith('csv')):
          scores_other = pandas.read_csv(OTHER_SCORES)
        else:
           oscore = open(OTHER_SCORES, "rb")
           scores_other = pickle.load(oscore)
           for mylabel in other_labels:
             scores_other[mylabel] = scores_other[mylabel].astype(float)

        #fig, ax = plt.subplots()
        #plot_AUPRC(scores_df, 'score', 'label', self.parameters["label1"], ax, colors[0], reverse_sign=1)
        #plot_AUPRC(scores_dmasif, 'avg_score', 'label', self.parameters["label3"], ax, colors[1], reverse_sign=0)
        #plot_AUPRC(scores_masif, 'score', 'label', self.parameters["label2"], ax, colors[8], reverse_sign=1)

        #other_labels = ['FIREDOCK', 'AP_PISA', 'CP_PIE', 'PYDOCK_TOT', 'ZRANK2', 'ROSETTADOCK', 'SIPPER']
        #pos_label = [0,0,1,0,0,0,1]
        #reverse_sign = [1,1,0,1,1,1,0]
        reverse_sign = PyIO.readSequential(PyPluMA.prefix()+"/"+self.parameters["pos_label"])




        for i in range(len(other_labels)):
           thresholds, matthews, metrics = find_optimal_threshold(scores_other, other_labels[i], 'label', reverse_sign=reverse_sign[i])
           all_results.append([other_labels[i]]+metrics)
           all_thresholds.append([other_labels[i]]+thresholds)
           all_matthews.append([other_labels[i]]+matthews)

        metrics_df = pandas.DataFrame(all_results, columns=['Method','BA','F1', 'Precision', 'Recall'])
        metrics_df.to_csv(outputfile, index=False)
        metrics_df

