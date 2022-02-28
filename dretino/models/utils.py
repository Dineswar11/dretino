import torch
import torch.nn as nn
import torch.nn.functional as F


def label_to_levels(label, num_classes, dtype=torch.float32):
    if not label <= num_classes - 1:
        raise ValueError('Class label must be smaller or '
                         'equal to %d (num_classes-1). Got %d.'
                         % (num_classes - 1, label))
    if isinstance(label, torch.Tensor):
        int_label = label.item()
    else:
        int_label = label

    levels = [1] * int_label + [0] * (num_classes - 1 - int_label)
    levels = torch.tensor(levels, dtype=dtype)
    return levels


def levels_from_labelbatch(labels, num_classes, dtype=torch.float32):
    levels = []
    for label in labels:
        levels_from_label = label_to_levels(
            label=label, num_classes=num_classes, dtype=dtype)
        levels.append(levels_from_label)

    levels = torch.stack(levels)
    return levels


def proba_to_label(probas):
    predict_levels = probas > 0.5
    predicted_labels = torch.sum(predict_levels, dim=1)
    return predicted_labels


def coral_loss(logits, levels, importance_weights=None, reduction='mean'):
    if not logits.shape == levels.shape:
        raise ValueError("Please ensure that logits (%s) has the same shape as levels (%s). "
                         % (logits.shape, levels.shape))

    term1 = (F.logsigmoid(logits) * levels
             + (F.logsigmoid(logits) - logits) * (1 - levels))

    if importance_weights is not None:
        term1 *= importance_weights

    val = (-torch.sum(term1, dim=1))

    if reduction == 'mean':
        loss = torch.mean(val)
    elif reduction == 'sum':
        loss = torch.sum(val)
    elif reduction is None:
        loss = val
    else:
        s = ('Invalid value for `reduction`. Should be "mean", '
             '"sum", or None. Got %s' % reduction)
        raise ValueError(s)

    return loss


def corn_loss(logits, y, num_classes):
    sets = []
    for i in range(num_classes - 1):
        label_mask = y > i - 1
        label_tensor = (y[label_mask] > i).to(torch.int64)
        sets.append((label_mask, label_tensor))

    num_examples = 0
    losses = 0.
    for task_index, s in enumerate(sets):
        train_examples = s[0]
        train_labels = s[1]

        if len(train_labels) < 1:
            continue

        num_examples += len(train_labels)
        pred = logits[train_examples, task_index]

        loss = -torch.sum(F.logsigmoid(pred) * train_labels
                          + (F.logsigmoid(pred) - pred) * (1 - train_labels))
        losses += loss

    return losses / num_classes


def corn_labels_from_logits(logits):
    probas = torch.sigmoid(logits)
    probas = torch.cumprod(probas, dim=1)
    predict = probas > 0.5
    preds = torch.sum(predict, dim=1)
    return preds


class CoralLayer(nn.Module):
    def __init__(self, size_in, num_classes, preinit_bias=True):
        super().__init__()
        self.size_in, self.size_out = size_in, 1

        self.coral_weights = torch.nn.Linear(self.size_in, 1, bias=False)
        if preinit_bias:
            self.coral_bias = torch.nn.Parameter(
                torch.arange(num_classes - 1, 0, -1).float() / (num_classes - 1))
        else:
            self.coral_bias = torch.nn.Parameter(
                torch.zeros(num_classes - 1).float())

    def forward(self, x):
        return self.coral_weights(x) + self.coral_bias
