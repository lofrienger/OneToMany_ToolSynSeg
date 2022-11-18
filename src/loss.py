from torch import nn


class LossBinary:
    def __init__(self):
        self.nll_loss = nn.BCEWithLogitsLoss()

    def __call__(self, outputs, targets):
        loss = self.nll_loss(outputs, targets)

        return loss
