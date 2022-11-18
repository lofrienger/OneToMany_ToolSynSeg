import numpy as np
import torch
from tqdm import tqdm

import utils


def validation_binary(args, model, criterion, valid_loader, num_classes=None):
    with torch.no_grad():
        model.eval()
        losses = []

        iou = []
        dice = []

        tq = tqdm(total=(len(valid_loader) * args.batch_size))
        tq.set_description('Valid: ')

        for _, (inputs, targets) in enumerate(valid_loader):
            inputs = utils.cuda(inputs)
            targets = utils.cuda(targets)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            losses.append(loss.item())

            iou_t = get_iou(targets, (outputs > 0).float())
            iou += iou_t
            # dice = (2 * iou) / (iou + 1)
            dice += [((2 * iou_t[i]) / (iou_t[i] + 1))
                     for i in range(len(iou_t))]
            update_size = inputs.size(0)
            tq.update(update_size)
        tq.close()

        valid_loss = np.mean(losses)  # type: float
        valid_iou = np.mean(iou).astype(np.float64)
        valid_dice = np.mean(dice).astype(np.float64)

        print('Valid loss: {:.5f}, iou: {:.5f}, dice: {:.5f}'.format(
            valid_loss, valid_iou, valid_dice))
        metrics = {'valid_loss': valid_loss,
                   'iou': valid_iou, 'dice': valid_dice}

        return metrics


def get_iou(y_true, y_pred):
    epsilon = 1e-15
    intersection = (y_pred * y_true).sum(dim=-2).sum(dim=-1)
    union = y_true.sum(dim=-2).sum(dim=-1) + y_pred.sum(dim=-2).sum(dim=-1)

    return list(((intersection + epsilon) / (union - intersection + epsilon)).data.cpu().numpy())
