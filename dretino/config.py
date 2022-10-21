# Random Seed
SEED = 42

# Num Workers
NUM_WORKERS = 4


# M0del Details
NUM_CLASSES = 5
ADDITIONAL_LAYERS = True
NUM_NEURONS = 512
N_LAYERS = 2
DROPOUT_RATE = 0.2

# Unlabeled Pretrained Model Links
DINO_LINK = "https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/dino_resnet50_pretrain_full_checkpoint.pth"

SWAV_LINK = "https://pl-bolts-weights.s3.us-east-2.amazonaws.com/swav/bolts_swav_imagenet/swav_imagenet.ckpt"

SIMCLR_LINK = "https://pl-bolts-weights.s3.us-east-2.amazonaws.com/simclr/bolts_simclr_imagenet/simclr_imagenet.ckpt"
