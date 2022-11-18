# import comet_ml at the top of your file
from comet_ml import Experiment
import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from albumentations import CoarseDropout, Compose, Normalize, Resize
from torch import nn
from torch.optim import Adam

from loss import LossBinary
from models import UNet, model_list
from utils import *
from validation import validation_binary
from segmix import FastCollateMixup

warnings.filterwarnings("ignore")

# assign GPU ID
# os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"

def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--seed', type=int, default=3407)
    arg('--batch_size', type=int, default=64)
    arg('--n_epochs', type=int, default=100, help='the number of total epochs')
    arg('--lr', type=float, default=1e-3)
    arg('--workers', type=int, default=8)
    arg('--num_classes', type=int, default=1)
    arg('--model', type=str, default='UNet', choices=model_list.keys())
    arg('--save_model', type=str, default='True', help='save model or not')

    arg('--train_dataset', type=str, default='Endo18_train', 
        choices=["Endo18_train", "Syn-S2-F1", "Syn-S2-F2", "Syn-S3-F1F2", "Syn-S3-F1", "Syn-S3-F2", \
                'Joint_Syn-S2-F1', 'Joint_Syn-S2-F2', 'Joint_Syn-S3-F1', 'Joint_Syn-S3-F2', 'Joint_Syn-S3-F1F2', \
                'Syn-S2-F1_inc_Real', 'Syn-S2-F2_inc_Real', 'Syn-S3-F1F2_inc_Real', 'Syn-S3-F1_inc_Real', 'Syn-S3-F2_inc_Real'], \
                help='training dataset')
    arg('--val_dataset', type=str, default='Endo18_test', choices=["Endo18_test", ], help='validation dataset')
    arg('--blend_mode', type=str, default='None', choices=["alpha", "gaussian", "laplacian", "None"])
    
    arg('--real_ratio', type=float, default=0.0)
    arg('--inc_ratio', type=float, default=0.0)

    arg('--augmix', type=str, default='None', choices=["None", "I", "II", "III", "IV"])
    arg('--augmix_level', type=int, default=0, choices=[0, 1, 2, 3])

    arg('--coarsedropout', type=str, default='None', choices=["None", "hole14_w13_h13_p5"])
    arg('--cutmix_collate', type=str, default='default_collate', choices=["default_collate", "FastCollateMixup"])

    arg('--comet_api_key', type=str, default='api_key', help='your comet api key')
    arg('--experiment', help='Comet experiment instance')

    args = parser.parse_args()

    # deterministic training
    seed_everything(args.seed)

    # set model saved path
    # model_path = set_model_saved_path(args)
    if args.train_dataset == 'Endo18_train':
        model_path = 'saved_model/' + args.train_dataset + '/' + args.val_dataset \
            + '/cut_'+args.coarsedropout + '/colloate_'+args.cutmix_collate
    elif 'Joint' in args.train_dataset:
        model_path = 'saved_model/joint_real_' + str(args.real_ratio) + '/' + args.train_dataset + '/' + args.val_dataset + '/' + args.blend_mode \
            + '/cut_'+args.coarsedropout + '/colloate_'+args.cutmix_collate
    elif 'inc' in args.train_dataset:
        model_path = 'saved_model/inc_' + str(args.inc_ratio) + '/' + args.train_dataset + '/' + args.val_dataset + '/' + args.blend_mode \
            + '/cut_'+args.coarsedropout + '/colloate_'+args.cutmix_collate
    else:
        model_path = 'saved_model/' + args.train_dataset + '/' + args.val_dataset + '/' + args.blend_mode \
            + '/cut_'+args.coarsedropout + '/colloate_'+args.cutmix_collate
    
    if args.augmix != 'None':
        model_path = model_path + '/augmix_' + args.augmix + '_L' + str(args.augmix_level)
    if args.save_model == 'True':
        if not os.path.isdir(model_path):
            Path(model_path).mkdir(parents=True, exist_ok=True)
        else:
            if os.path.exists(model_path + '/best_model_dice.pt'):
                print("==> WARNING: The same model path exists and model have been saved!")
                sys.exit()
    else:
        print('==> Model will not be saved.')


    # Create an experiment with your api key
    disable_comet = (args.save_model != 'True')
    print("==> Save experiment to Comet:", (not disable_comet))
    args.experiment = Experiment(api_key=args.comet_api_key, project_name="OneToMany_ToolSynSeg", workspace="WS", disabled=disable_comet)
    
    if args.train_dataset == 'Endo18_train':
        args.experiment.set_name(f'train_dataset_{args.train_dataset}-val_dataset_{args.val_dataset}-augmix_{args.augmix}-L{args.augmix_level}')
    elif 'inc' in args.train_dataset:
        args.experiment.set_name(f'train_dataset_{args.train_dataset}-inc_ratio_{args.inc_ratio}-mode_{args.blend_mode}-val_dataset_{args.val_dataset}-augmix_{args.augmix}-L{args.augmix_level}')
    else:
        args.experiment.set_name(f'train_dataset_{args.train_dataset}-real_ratio_{args.real_ratio}-mode_{args.blend_mode}-val_dataset_{args.val_dataset}-augmix_{args.augmix}-L{args.augmix_level}')

    if args.coarsedropout != 'None':
        args.experiment.add_tag(f'cut_{args.coarsedropout}')
    if args.cutmix_collate != 'default_collate':
        args.experiment.add_tag(f'collate_{args.cutmix_collate}')

    if args.save_model == 'True':
        print('Model will be saved to:', model_path)

    print('=====================================')
    print('model            :', args.model)
    print('train_dataset    :', args.train_dataset)
    print('real_ratio       :', args.real_ratio)
    print('inc_ratio        :', args.inc_ratio)
    print('val_dataset      :', args.val_dataset)
    print('blend_mode       :', args.blend_mode)
    print('augmix           :', args.augmix)
    print('augmix_level     :', args.augmix_level)
    print('coarsedropout    :', args.coarsedropout)
    print('cutmix_collate   :', args.cutmix_collate)
    print('=====================================')
    
    # initialize model
    if args.model == 'UNet':
        model = UNet(num_classes=args.num_classes)
    else:
        model_name = model_list[args.model]
        model = model_name(num_classes=args.num_classes, pretrained=True)

    model = model.cuda()  # put model weights into GPU

    # assign GPU device
    if torch.cuda.is_available():
        num_gpu = torch.cuda.device_count()
        print('The total available GPU:', num_gpu)
        if num_gpu > 1:  # has more than 1 gpu
            device_ids = np.arange(num_gpu).tolist()
            model = nn.DataParallel(model, device_ids=device_ids).cuda()
    else:
        raise SystemError('GPU device not found')

    # loss function
    loss = LossBinary()
    # optimizer
    optimizer = Adam(model.parameters(), lr=args.lr)
    # validation function
    valid = validation_binary
    
    Syn_root = Path('/mnt/data-ssd/wa/miccai_ext/data_gen/syndata')
    endo18_root_path = Path('/mnt/data-hdd/wa/dataset/EndoVis/2018_RoboticSceneSegmentation/')  
    endo18_test_path = endo18_root_path / 'test' # Path('/mnt/data-hdd/wa/dataset/EndoVis/2018_RoboticSceneSegmentation/test/')

    endo18_isinet_path = endo18_root_path / 'ISINet_Train_Val'
    endo18_train_path = endo18_isinet_path / 'train'
    endo18_val_path = endo18_isinet_path / 'val'
    endo18_train_image_paths = list((endo18_train_path / 'images').glob('*')) + list((endo18_val_path / 'images').glob('*')) # 2235
    endo18_train_len = len(endo18_train_image_paths)
    print('Number of dataset endo18 training images:', endo18_train_len)
    
    train_image_paths, val_image_paths, test_image_paths = [], [], []

    if args.train_dataset == 'Endo18_train':
        train_image_paths = endo18_train_image_paths # 2235

    elif args.train_dataset in ['Syn-S2-F1', 'Syn-S2-F2', 'Syn-S3-F1', 'Syn-S3-F2']:        
        Syn_image_paths = list((Syn_root / args.train_dataset / args.blend_mode / 'images').glob('*'))
        print('Original number of dataset images:', len(Syn_image_paths))
        np.random.shuffle(Syn_image_paths)
        train_image_paths = Syn_image_paths[:endo18_train_len]
        print('Selected number of dataset images:', len(train_image_paths))

    elif args.train_dataset == 'Syn-S3-F1F2': # combine 20% Syn-S3-F1 and 80% Syn-S3-F2
        Syn_S3_F1_image_paths = list((Syn_root / 'Syn-S3-F1' / args.blend_mode / 'images').glob('*'))
        print('Original number of dataset Syn-S3-F1 images:', len(Syn_S3_F1_image_paths))
        np.random.shuffle(Syn_S3_F1_image_paths)
        num_Syn_S3_F1_sel = int(0.2 * endo18_train_len)
        train_image_paths = Syn_S3_F1_image_paths[:num_Syn_S3_F1_sel]
        print('Selected number of dataset Syn-S3-F1 images:', num_Syn_S3_F1_sel)
        # print(len(train_image_paths))
        Syn_S3_F2_image_paths = list((Syn_root / 'Syn-S3-F2' / args.blend_mode / 'images').glob('*'))
        print('Original number of dataset Syn-S3-F2 images:', len(Syn_S3_F2_image_paths))
        np.random.shuffle(Syn_S3_F2_image_paths)
        num_Syn_S3_F2_sel = endo18_train_len-num_Syn_S3_F1_sel
        train_image_paths = train_image_paths + Syn_S3_F2_image_paths[:num_Syn_S3_F2_sel]
        print('Selected number of dataset Syn-S3-F2 images:', num_Syn_S3_F2_sel)

    elif args.train_dataset in ['Joint_Syn-S2-F1', 'Joint_Syn-S2-F2', 'Joint_Syn-S3-F1', 'Joint_Syn-S3-F2', 'Joint_Syn-S3-F1F2']:
        num_real = int(args.real_ratio * endo18_train_len)
        num_syn = endo18_train_len - num_real
        Syn_dataset = (args.train_dataset).split('_')[-1]
        if Syn_dataset in ['Syn-S2-F1', 'Syn-S2-F2', 'Syn-S3-F1', 'Syn-S3-F2']:
            ## Syn-S2-F1 / Syn-S2-F2 / Syn-S3-F1 / Syn-S3-F2
            Syn_image_paths = list((Syn_root / Syn_dataset / args.blend_mode / 'images').glob('*'))
            np.random.shuffle(Syn_image_paths)
            train_image_paths = train_image_paths + Syn_image_paths[:num_syn]
            ##
        elif Syn_dataset == 'Syn-S3-F1F2':
            ## Syn-S3-F1F2_image_paths
            Syn_S3_F1_image_paths = list((Syn_root / 'Syn-S3-F1' / args.blend_mode / 'images').glob('*'))
            np.random.shuffle(Syn_S3_F1_image_paths)
            num_Syn_S3_F1_sel = int(0.2 * num_syn)
            train_image_paths = train_image_paths + Syn_S3_F1_image_paths[:num_Syn_S3_F1_sel]
            Syn_S3_F2_image_paths = list((Syn_root / 'Syn-S3-F2' / args.blend_mode / 'images').glob('*'))
            np.random.shuffle(Syn_S3_F2_image_paths)
            num_Syn_S3_F2_sel = num_syn-num_Syn_S3_F1_sel
            train_image_paths = train_image_paths + Syn_S3_F2_image_paths[:num_Syn_S3_F2_sel]
            ##
        np.random.shuffle(endo18_train_image_paths)
        train_image_paths = train_image_paths + endo18_train_image_paths[:num_real]
    elif 'inc_Real' in args.train_dataset:
        # Syn part
        Syn_dataset = (args.train_dataset).split('_')[0]
        if Syn_dataset in ['Syn-S2-F1', 'Syn-S2-F2', 'Syn-S3-F1', 'Syn-S3-F2']:
            ## Syn-S2-F1 / Syn-S2-F2 / Syn-S3-F1 / Syn-S3-F2
            Syn_image_paths = list((Syn_root / Syn_dataset / args.blend_mode / 'images').glob('*'))
            np.random.shuffle(Syn_image_paths)
            train_image_paths = train_image_paths + Syn_image_paths[:endo18_train_len]
            ##
        elif Syn_dataset == 'Syn-S3-F1F2':
            ## Syn-S3-F1F2_image_paths
            Syn_S3_F1_image_paths = list((Syn_root / 'Syn-S3-F1' / args.blend_mode / 'images').glob('*'))
            np.random.shuffle(Syn_S3_F1_image_paths)
            num_Syn_S3_F1_sel = int(0.2 * endo18_train_len)
            train_image_paths = train_image_paths + Syn_S3_F1_image_paths[:num_Syn_S3_F1_sel]
            Syn_S3_F2_image_paths = list((Syn_root / 'Syn-S3-F2' / args.blend_mode / 'images').glob('*'))
            np.random.shuffle(Syn_S3_F2_image_paths)
            num_Syn_S3_F2_sel = endo18_train_len-num_Syn_S3_F1_sel
            train_image_paths = train_image_paths + Syn_S3_F2_image_paths[:num_Syn_S3_F2_sel]
            ##

        # Real part
        np.random.shuffle(endo18_train_image_paths)
        num_real = int(endo18_train_len * args.inc_ratio)
        train_image_paths = train_image_paths + endo18_train_image_paths[:num_real]

    if args.val_dataset == 'Endo18_test':
        val_image_paths = list((endo18_test_path / 'images').glob('*')) # 999
    else:
        val_image_paths = []
    
    test_image_paths = list((endo18_test_path / 'images').glob('*')) # 999

    # remove 2 images which don't have annotations
    bug_images = [Path('/mnt/data-hdd/wa/dataset/EndoVis/2018_RoboticSceneSegmentation/test/images/seq_3_frame249.png'),
                  Path('/mnt/data-hdd/wa/dataset/EndoVis/2018_RoboticSceneSegmentation/test/images/seq_4_frame249.png')]
    for bug_image in bug_images:
        if bug_image in val_image_paths:
            val_image_paths.remove(bug_image)
        if bug_image in test_image_paths:
            test_image_paths.remove(bug_image)

    assert len(train_image_paths) != 0, 'invalid train_image_paths'
    assert len(val_image_paths) != 0, 'invalid val_image_paths'
    assert len(test_image_paths) != 0, 'invalid test_image_paths'
    print('Num train = {}, Num_val = {}, Num test = {}'.format(len(train_image_paths), len(val_image_paths), len(test_image_paths)))

    args.experiment.log_parameters(vars(args))

    def train_transform_cdo(p=1):
        return Compose([
            Resize(224, 224, always_apply=True, p=1),
            CoarseDropout(max_holes=4, min_holes=1, max_height=0.3, max_width=0.3, min_height=0.1, min_width=0.1, mask_fill_value=0, p=0.5),
            Normalize(p=1)
        ], p=p)

    def train_transform_augmix_cdo(p=1):
        return Compose([
            Resize(224, 224, always_apply=True, p=1),
            CoarseDropout(max_holes=4, min_holes=1, max_height=0.3, max_width=0.3, min_height=0.1, min_width=0.1, mask_fill_value=0, p=0.5)
        ], p=p)

    def make_loader_cut(args, file_names, shuffle=False, transform=None, mode='train'):
        dataset=EndoDataset(args, file_names, transform=transform, mode=mode)

        if args.augmix != 'None' and mode == 'train': 
            dataset = AugMixData(dataset, preprocess, aug_type = args.augmix, level=args.augmix_level)

        cutmix_args = {
            'mixup_alpha': 0.,
            'cutmix_minmax': (0.3, 0.8),
            'prob': 0.5,
            'mode': 'elem',
        }

        if args.cutmix_collate == 'FastCollateMixup' and mode=='train':
            collator = FastCollateMixup(**cutmix_args)
        else:
            collator = torch.utils.data.dataloader.default_collate

        return DataLoader(
            dataset=dataset,
            shuffle=shuffle,
            num_workers=args.workers,
            batch_size=args.batch_size,
            pin_memory=torch.cuda.is_available(), collate_fn=collator
        )

    if args.coarsedropout == 'None':
        if args.augmix == 'None':
            train_loader = make_loader_cut(args, train_image_paths, shuffle=True, transform=train_transform(p=1), mode='train')
        else:
            train_loader = make_loader_cut(args, train_image_paths, shuffle=True, transform=train_transform_augmix(p=1), mode='train')
    else:
        if args.augmix == 'None':
            train_loader = make_loader_cut(args, train_image_paths, shuffle=True, transform=train_transform_cdo(p=1), mode='train')
        else:
            train_loader = make_loader_cut(args, train_image_paths, shuffle=True, transform=train_transform_augmix_cdo(p=1), mode='train')
    
    valid_loader = make_loader_cut(args, val_image_paths, transform=val_transform(p=1), mode='val')

    with args.experiment.train():
        train(
            args=args,
            model=model,
            criterion=loss,
            train_loader=train_loader,
            valid_loader=valid_loader,
            validation=valid,
            optimizer=optimizer,
            model_path=model_path
        )

    if args.save_model == 'True':
        with args.experiment.test():
            print('==> Test begin ▶️')

            endo18_images_paths, endo17_images_paths, RoboTool_images_paths = get_dataset_paths()

            checkpoint = torch.load(os.path.join(model_path, 'best_model_dice.pt'))
            model.load_state_dict(checkpoint['net'])  # load the model's parameters
            with torch.no_grad():
                model.eval()

                print('1. Test on endo18 ==>')
                test_loader_endo18 = make_loader_cut(args, endo18_images_paths, transform=test_transform(p=1), mode='eva')
                test_metrics = validation_binary(args, model=model, criterion=loss, valid_loader=test_loader_endo18)
                log_dict = {'endo18 dice': round(test_metrics['dice']*100, 2), 'endo18 iou': round(test_metrics['iou']*100, 2)}
                args.experiment.log_others(log_dict)
                print('Test result:', log_dict)

                print('2. Test on endo17 ==>')
                test_loader_endo17 = make_loader_cut(args, endo17_images_paths, transform=test_transform(p=1), mode='eva')
                test_metrics = validation_binary(args, model=model, criterion=loss, valid_loader=test_loader_endo17)
                log_dict = {'endo17 dice': round(test_metrics['dice']*100, 2), 'endo17 iou': round(test_metrics['iou']*100, 2)}
                args.experiment.log_others(log_dict)
                print('Test result:', log_dict)

                print('3. Test on RoboTool ==>')
                test_loader_RoboTool = make_loader_cut(args, RoboTool_images_paths, transform=test_transform(p=1), mode='eva')
                test_metrics = validation_binary(args, model=model, criterion=loss, valid_loader=test_loader_RoboTool)
                log_dict = {'RoboTool dice': round(test_metrics['dice']*100, 2), 'RoboTool iou': round(test_metrics['iou']*100, 2)}
                args.experiment.log_others(log_dict)
                print('Test result:', log_dict)                 
                
                print('==> Test finish ⏹️')

if __name__ == '__main__':
    main()
