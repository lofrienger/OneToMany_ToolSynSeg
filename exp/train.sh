## Sample training commands

# 1. train on Endo18 dataset
CUDA_VISIBLE_DEVICES=0,1 python src/train.py --train_dataset Endo18_train --augmix I --augmix_level 2 --coarsedropout hole14_w13_h13_p5 --cutmix_collate FastCollateMixup

# 2. train on Synthetic dataset
CUDA_VISIBLE_DEVICES=0,1 python src/train.py --train_dataset Syn-S3-F1F2 --blend_mode alpha --augmix I --augmix_level 2 --coarsedropout hole14_w13_h13_p5 --cutmix_collate FastCollateMixup

# 3. joint train: 90% Syn-S3-F1F2 + 10% Endo18
CUDA_VISIBLE_DEVICES=0,1 python src/train.py --train_dataset Joint_Syn-S3-F1F2 --blend_mode alpha --real_ratio 0.1 --augmix I --augmix_level 2 --coarsedropout hole14_w13_h13_p5 --cutmix_collate FastCollateMixup

# 4. joint train: Syn-S3-F1F2 + 10% Endo18
CUDA_VISIBLE_DEVICES=0,1 python src/train.py --train_dataset Syn-S3-F1F2_inc_Real --blend_mode alpha --inc_ratio 0.1 --augmix I --augmix_level 2 --coarsedropout hole14_w13_h13_p5 --cutmix_collate FastCollateMixup
