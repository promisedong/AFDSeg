from glob import glob
import os
import random

# -------------------------------------------------------#
#   生成训练和测试文件
#   拆分比例为 8:2
#--------------------------------------------------------#

def Generate(path = None,
             datasetsname = None, #数据集名称
             suffix = '.jpg',
             ratio = 0.8 #split train & val
             ):

    # 训练文件
    traintxt = f'{datasetsname}_train.txt'
    # 测试文件
    valtext = f'{datasetsname}_val.txt'

    # 遍历所有数据
    images = os.listdir(os.path.join(path,f'images'))
    labels = os.listdir(os.path.join(path,f'masks'))

    assert len(images)==len(labels),'train image not equal train label'

    # shuffle train datasets

    random.shuffle(images)

    _ = int(len(images) * ratio)

    train_image = random.sample(images,_)

    traintxt = open(traintxt,'w+')
    valtxt = open(valtext,'w+')

    # train image
    for _d in images:
        imagepath = os.path.join(path,f'images',_d)
        maskpath = os.path.join(path,f'masks',_d)

        if (_d not  in train_image):

            valtxt.write(imagepath+' '+maskpath+'\n')
        else:
         traintxt.write(imagepath + ' ' + maskpath + '\n')


    traintxt.close()
    valtxt.close()
    print('Generate train and val file finished......')


#-------------------------------------------------------#
#   convert data format
#-------------------------------------------------------#
def convert_data_format(path,
                        name,
                        datasetname = None):

    txt = os.path.join(path,name)

    file = open(txt,'r').readlines()

    mode = name.split('.')[0]
    wfile = open(datasetname+'_'+mode+'.txt','w+')

    for f in file:
        f = f.strip()
        jpgpath = os.path.join(path,'images',f+'.jpg')
        maskpath = os.path.join(path,'masks',f+'.png')

        wfile.write(jpgpath+' '+maskpath+'\n')





if __name__ == '__main__':
   path = '/root/data1/Gasking_Segmentation/datasets/kvasir-seg'

   Generate(path = path,datasetsname = 'kvasir_seg')

   # path = r'D:\JinKuang\24_8_25_Gasking\datasets\kvasir-instrument'
   # convert_data_format(path, 'train.txt','kvasir_instrument')
   # convert_data_format(path, 'test.txt','kvasir_instrument')
   #
   # path = r'D:\JinKuang\24_8_25_Gasking\datasets\Kvasir-Capsule'
   #
   # Generate(path=path, datasetsname='kvasir_Capsule')


