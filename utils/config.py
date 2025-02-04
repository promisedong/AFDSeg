#--------------------------------#
# 配置文件
#--------------------------------#

Config = {

    'num_worker':0,
    'decay_rate':1e-3, # if use Adam
    'adam_max_lr': 2.5e-4,
    'sgd_max_lr':2.5e-4,
    'min_lr': 1e-6,
    'momentum':0.,
    'optimizer':'adam',

    'init_epoch':0,
    'freeze_epoch': 15,
    'unfreeze_epoch':50,

    # train 15 50
    # uda 5 15

    'bs':1,

    'mode':'co-train',
    #'mode':'rank',



    '_Class':('polyp ',), #使用元组为一个类别时，记得加上逗号

    'pretrained': True,
    'model_path':'model_data/18.pth',
    'save_step':1,
    'input_shape':(224,224), #输入尺度

    'train_txt': r'train.txt',
    'test_txt': r'test.txt',
    'val_txt': r'val.txt',

    'mean':[ 0.485, 0.456, 0.406 ],
    'std':[ 0.229, 0.224, 0.225 ],

    'platte':[[191,246,195],[200,210,240]]  #二分类
}