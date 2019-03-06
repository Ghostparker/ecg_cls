import torch


cls_test = {
    'num_classes' : 5,
    'trainset_root' : '/train/results/ecg_cls/',
    'train_images_set' : (('index_dataset','trainlist.txt'),),
    'testset_root' : '/train/results/ecg_cls/',
    'test_images_set' : (('index_dataset','testlist.txt'),),
    'result_savepath' : './results/',
    'error_savepath' : './errorfile/',


    'train_batch_size' : 256,
    'test_batch_size' : 256,
    'lr_change' : 'lr_steps',
    'learningrate' : 1e-3,
    'lr_steps' :[2000,4000,6000],
    'gamma' : 0.2,

    'start_iter' : 0,
    'max_iter' : 8000,
    'log_iter' : 100,
    'save_iter' : 1000,


    'use_gpu' : True,
    'device_ids' : [0,],
    'gpu_id' : 0,

    'is_pretrained' : False,
    'pretrained_path' : './weight/f1.pth',
    'eval_model_path' : './usemodel/model_40000.pth',
    'base_model' : 'ResNet18',
    'save_model_dir' : './weight',
}
