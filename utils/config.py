import torch


cls_test = {
    'num_classes' : 5,
    'trainset_root' : '/train/results/ecg_cls/',
    'train_images_set' : (('index_dataset','trainlist.txt'),),
    'testset_root' : '/train/results/ecg_cls/',
    'test_images_set' : (('index_dataset','testlist.txt'),),

    'train_batch_size' : 128,
    'test_batch_size' : 128,
    'lr_change' : 'cosine',
    'learningrate' : 1e-3,
    'lr_steps' :(1000,2000,3000),
    'gamma' : 0.1,

    'start_iter' : 0,
    'max_iter' : 4000,
    'log_iter' : 10,
    'save_iter' : 100,


    'use_gpu' : True,
    'device_ids' : [0,],
    'gpu_id' : 0,

    'is_pretrained' : False,
    'pretrained_path' : './weight/f1.pth',
    'eval_model_path' : './weight/model_4000.pth',
    'base_model' : 'ResNet18',
    'save_model_dir' : './weight',
}
