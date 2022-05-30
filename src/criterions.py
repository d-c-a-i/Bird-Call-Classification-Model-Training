import torch
import torch.nn as nn

from torch.nn import BCEWithLogitsLoss


def bce_loss_with_logits(y_preds, y_true):
    bce_loss = BCEWithLogitsLoss()

    if not torch.is_tensor(y_preds):
        y_preds = torch.Tensor(y_preds)

    if not torch.is_tensor(y_true):
        y_true = torch.Tensor(y_true)

    with torch.no_grad():
        return bce_loss(y_preds, y_true)


# More clear version, From https://www.kaggle.com/c/rfcx-species-audio-detection/discussion/213075
class FocalLoss(nn.Module):  # gamma=0, alpha=0.75 or gamma=2, alpha=0.25
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, preds, targets):
        loss_fct = nn.BCEWithLogitsLoss(reduction='none')
        bce_loss = loss_fct(preds, targets)
        probas = torch.sigmoid(preds)
        loss = torch.where(targets >= 0.1, self.alpha * ((1. - probas)**self.gamma) * bce_loss,
                           (1. - self.alpha) * (probas**self.gamma) * bce_loss)
        loss = loss.mean()

        return loss


class BirdBCEwLogitsLoss(nn.Module):
    def __init__(self, class_weights=None, label_smoothing=0.0):
        super(BirdBCEwLogitsLoss, self).__init__()

        self.class_weights = class_weights
        self.smoothing = label_smoothing
        self.bce_loss = BCEWithLogitsLoss(reduction='none')

    @staticmethod
    def _smooth(targets: torch.Tensor, smoothing=0.0, sec_labels_mask=None):
        if smoothing == 0.0:
            return targets

        assert(0 <= smoothing < 1)
        with torch.no_grad():
            # targets = targets*(1.0 - smoothing) + 0.5*smoothing  # if smoothing=0.01, target 0 -> 0.005, 1 -> 0.995
                                                                 # if smoothing=0.2, target 0 -> 0.1, 1 -> 0.9
            targets[sec_labels_mask == 1] = targets[sec_labels_mask == 1] * (1.0 - smoothing) + 0.5 * smoothing

        return targets

    def forward(self, preds, targets, sec_labels_mask=None):
        targets = BirdBCEwLogitsLoss._smooth(targets, self.smoothing, sec_labels_mask)

        bce_loss = self.bce_loss(preds, targets)  # bce_loss: (bs, 397)
        if self.class_weights is not None:
            bce_loss = bce_loss * self.class_weights

        # if sec_labels_mask is not None:
        #     bce_loss = bce_loss[sec_labels_mask != 1]  # mask out sec_labels in loss

        loss = bce_loss.mean()
        return loss
