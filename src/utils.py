import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import tqdm

from dataset import *


def seed_everything(seed=3407):
    '''set seed for deterministic training'''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def cuda(x):
    return x.cuda(non_blocking=True) if torch.cuda.is_available() else x


def str_insert(str_origin, pos, str_add):
    str_list = list(str_origin)
    str_list.insert(pos, str_add)
    str_out = ''.join(str_list)
    return str_out


def train_test_split(num, val_ratio):
    indices = list(range(num))
    split = int(np.floor(val_ratio * num))
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    return train_indices, val_indices


def write_event(log, step, **data):
    data['step'] = step
    data['dt'] = datetime.now().isoformat()
    log.write(json.dumps(data, sort_keys=True))
    log.write('\n')
    log.flush()


def check_crop_size(image_height, image_width):
    """Checks if image size divisible by 32.

    Args:
        image_height:
        image_width:

    Returns:
        True if both height and width divisible by 32 and False otherwise.

    """
    return image_height % 32 == 0 and image_width % 32 == 0


def get_time_suffix():
    ''' get current time, return string in format: year_month_day, e.g., 2021_11_11 '''
    time_struct = time.localtime(time.time())
    year, month, day = time_struct.tm_year, time_struct.tm_mon, time_struct.tm_mday
    time_suffix = str(year) + '_' + str(month) + '_' + str(day)
    return time_suffix


def set_model_saved_path(args):
    if args.train_dataset == 'Endo18_train':
        save_model_path = 'saved_model/' + args.method + '/' + args.train_dataset + '/' + args.val_dataset + '/seed_' + str(args.seed)
    else:
        save_model_path = 'saved_model/' + args.method + '/' + args.train_dataset + '/' + args.val_dataset + '/' + args.blend_mode + '/seed_' + str(args.seed)
    
    if args.augmix != 'None':
        save_model_path = save_model_path + '/augmix_' + args.augmix + '_L' + str(args.augmix_level)

    if args.save_model == 'True':
        if not os.path.isdir(save_model_path):
            print('==> Model will be saved to:', save_model_path)
            Path(save_model_path).mkdir(parents=True, exist_ok=True)
        else:
            if os.path.exists(save_model_path + '/best_model_dice.pt'):
                print("==> WARNING: The same model path exists and model have been saved!")
                sys.exit()
    else:
        print('==> Model will not be saved.')
    
    return save_model_path


def train(args, model, criterion, train_loader, valid_loader, validation, optimizer, model_path):

    valid_losses = []
    train_losses = []
    save_model_path = model_path
    best_dice, best_epoch_dice = 0.0, 0
    start_epoch = 0

    print('==> Training started.')

    for epoch in range(start_epoch + 1, args.n_epochs + 1):
        model.train()

        tq = tqdm.tqdm(total=(len(train_loader) * args.batch_size))
        tq.set_description('Epoch {}'.format(epoch))
        for _, (inputs, targets) in enumerate(train_loader):

            inputs = cuda(inputs)
            with torch.no_grad():
                targets = cuda(targets)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            train_losses.append(loss.item())

            optimizer.zero_grad()
            batch_size = inputs.size(0)
            loss.backward()
            optimizer.step()

            tq.update(batch_size)
        tq.close()
        tr_loss = np.mean(train_losses)
        args.experiment.log_metric('tr_loss', tr_loss, step=epoch)

        # ========================== Validation ========================== #
        valid_metrics = validation(args, model, criterion, valid_loader)
        valid_loss = valid_metrics['valid_loss']
        valid_losses.append(valid_loss)
        valid_iou = valid_metrics['iou']
        valid_dice = valid_metrics['dice']
        args.experiment.log_metrics(valid_metrics, step=epoch)

        checkpoint = {
            "net": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "valid_iou": valid_iou,
            "valid_dice": valid_dice,
            "best_dice": best_dice,
            "best_epoch_dice": best_epoch_dice,
            "valid_loss": valid_loss
        }

        if valid_dice > best_dice:
            print('=================== New best model of dice! ========================')
            best_dice = valid_dice
            best_epoch_dice = epoch
            if args.save_model == 'True':
                torch.save(checkpoint, os.path.join(
                    save_model_path, "best_model_dice.pt"))

        print('Best epoch for dice unitl now:',
              best_epoch_dice, ' best dice:', best_dice)

    print('==> Training finished.')
    print('Best epoch dice:', best_epoch_dice,
          ' best dice:', best_dice)
    args.experiment.log_others(
        {'best dice': best_dice, 'best dice epoch': best_epoch_dice})
