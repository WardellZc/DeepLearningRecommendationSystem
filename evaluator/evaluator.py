# from sklearn.metrics import *
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score


class Evaluator:
    def __init__(self):
        self.array = []

    @staticmethod
    def eval(y_true, y_pred):
        y_true = y_true.numpy()
        y_pred = y_pred.detach().numpy()
        y_pred = (y_pred >= 0.5).astype(int)
        array = [accuracy_score(y_true, y_pred), precision_score(y_true, y_pred), recall_score(y_true, y_pred),
                 f1_score(y_true, y_pred), roc_auc_score(y_true, y_pred)]
        return array
