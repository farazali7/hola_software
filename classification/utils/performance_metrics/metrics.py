from sklearn.metrics import accuracy_score, recall_score


def get_accuracy(labels, preds):
    return accuracy_score(labels, preds)


def get_specificity(labels, preds, pos_label=5):
    return recall_score(labels, preds, pos_label=pos_label)


def get_sensitivity(labels, preds, pos_label=5):
    return recall_score(labels, preds, pos_label=pos_label)