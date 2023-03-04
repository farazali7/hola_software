from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score


def get_accuracy(labels, preds):
    return accuracy_score(labels, preds)


def get_specificity(labels, preds, pos_label=0):
    return recall_score(labels, preds, pos_label=pos_label)


def get_sensitivity(labels, preds, pos_label=1):
    return recall_score(labels, preds, pos_label=pos_label)


def get_f1score(labels, preds, pos_label=1):
    return f1_score(labels, preds, pos_label=pos_label)


def get_precision(labels, preds, pos_label=1):
    return precision_score(labels, preds, pos_label=pos_label)
