from easydict import EasyDict as edict


def get_config(dataset):
    config = edict()
    config.dataset = dataset
    config.embedding_size = 512
    config.sample_rate = 1
    config.fp16 = False
    config.momentum = 0.9
    config.weight_decay = 5e-4
    config.batch_size = 64
    config.lr = 0.1
    config.output = "../work_dirs"
    if config.dataset == "ms1m" or config.dataset == "MS1M":
        config.rec = '/data2/zhouyang/dataset/FaceX/faces_emore'
        config.val_root = "/data2/zhouyang/dataset/FaceX/faces_emore"
        config.num_classes = 85742
        config.num_image = 'forget'
        config.num_epoch = 24
        config.warmup_epoch = -1
        config.val_targets = ["lfw", "cfp_fp", "agedb_30"]
        # config.val_targets = []
        def lr_step_func(epoch):
            return ((epoch + 1) / (4 + 1)) ** 2 if epoch < -1 else 0.1 ** len(
                [m for m in [5, 10, 15, 20] if m - 1 <= epoch])
        config.lr_func = lr_step_func

    return config
