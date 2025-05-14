
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import os
import yaml
from PIL import Image
import numpy as np
import json as jsonmod
from ipdb import set_trace
import random
from tqdm import tqdm,trange
#from  retrieval import parser_options
from config import parser_options               #导入配置文件

import numpy as np
import cv2
from PIL import Image
from pylab import*
import argparse
# os.environ["CUDA_VISIBLE_DEVICES"] = '3'
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class FlickrDataset(data.Dataset):
    """
    Dataset loader for Flickr30k and Flickr8k full datasets.
    """

    def __init__(self, root, json, split, transform=None,ids_=None):
        self.root = root
        self.split = split     #划分数据集
        self.transform = transform
        self.dataset = jsonmod.load(open(json, 'r'))['images']    #从json文件中加载数据集
        self.ids = []         #存储图像和对应句子的ID列表
       
        #遍历json文件中所有图像数据，根据split参数筛选参数
        for i, d in enumerate(self.dataset):
            if d['split'] == split:
                #五个句子
                '''默认情况下，每个图像对应5个句子'''
                self.ids += [(i, x) for x in range(len(d['sentences']))]
                #一个句子   
                '''可以选择只使用一个随机句子'''
                # self.ids += [(i, random.randint(0, len(d['sentences'])-1))]
                # self.ids += [(i, x) for x in range(len(d['sentences']))]

        #print(f"加载数据集: {len(self.dataset)} 条记录")
        #print(f"数据集 {split} 部分的 ID 数量: {len(self.ids)}")

    
    #根据索引返回一个数据项，包括图像、对应的句子和图像ID
    def __getitem__(self, index):
        """This function returns a tuple that is further passed to collate_fn
        """

        root = self.root
        ann_id = self.ids[index]     #获取索引对应的ID
        img_id = ann_id[0]           #获取图像ID
        cap_id = ann_id[1]           #获取句子ID
        caption = self.dataset[img_id]['sentences'][cap_id]['raw']         #获取句子
        # set_trace()
        path = self.dataset[img_id]['filename']     

        # 确保路径中没有多余的字符
        path = path.split(',')[0]    #加上的

        image = Image.open(os.path.join(root, path)).convert('RGB')    #打开图像文件并转换为RGB格式
        if self.transform is not None:
            image = self.transform(image)


        return image, caption ,img_id

    def __len__(self):
        return len(self.ids)              #返回数据集长度

#将一组 (图像, 标注) 元组构建成小批量张量
def collate_fn(data):
    """Build mini-batch tensors from a list of (image, caption) tuples.
    Args:
        data: list of (image, caption) tuple.
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length
    #按照标注的长度对数据列表进行排序，确保最长的标注在前
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images,captions,iids = zip(*data)    #将数据列表解压成图像、标注和图像ID的元组

    # Merge images (convert tuple of 3D tensor to 4D tensor)
    images = torch.stack(images, 0)   #合并图像

    return images,captions,iids

 

def get_loader_single(split, root, json, transform,
                      batch_size=100, shuffle=True,
                      num_workers=0, ids=None, collate_fn=collate_fn):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    #创建数据集
    dataset = FlickrDataset(root=root,
                            split=split,
                            json=json,
                            transform=transform,
                            ids_=ids)
    print("-------------------- "+ split + ": " + str(len(dataset)) + " ------------------------------")
    # Data loader   数据集加载器
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              pin_memory=True,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    return data_loader

#图像预处理
def get_transform(split_name):
    normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])   #标准化
    t_list = []            #初始化变换列表
   #t_list = [transforms.Resize(224)]
    t_list = [transforms.Resize((224, 224))]  # 统一图像尺寸
    t_end = [transforms.ToTensor(), normalizer]
    transform = transforms.Compose(t_list + t_end)            #组合变化
    return transform


#创建并返回数据加载器
def get_loaders(batch_size, opt):
    # Build Dataset Loader

    transform = get_transform('train')
    train_loader = get_loader_single( 'train',
                                        opt["dataset"]["data_image"],
                                        opt["dataset"]["data_json"],
                                        transform,
                                        batch_size=batch_size, shuffle=True,
                                        collate_fn=collate_fn)

    transform = get_transform('val')
    val_loader = get_loader_single( 'val',
                                    opt["dataset"]["data_image"],
                                    opt["dataset"]["data_json"],
                                    transform,
                                    batch_size=batch_size, shuffle=False,
                                    collate_fn=collate_fn)
    transform = get_transform('test')
    test_loader = get_loader_single( 'test',
                                    opt["dataset"]["data_image"],
                                    opt["dataset"]["data_json"],
                                    transform,
                                    batch_size=1, shuffle=False,
                                    collate_fn=collate_fn)
    if opt["dataset"]["data_json"].split("/")[-2] == "RSITMD":
        # set_trace()
        val_loader = test_loader
    return train_loader, val_loader, test_loader

'''
if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = '3'
    options = parser_options()     #解析配置文件
    #获取数据加载器
    train_loader, val_loader, test_loader = get_loaders(options["dataset"]["batch_size"], options)
    
     # 输出整个数据集的数量
    total_count = len(train_loader.dataset) + len(val_loader.dataset) + len(test_loader.dataset)
    print(f"整个数据集的数量: {total_count}")

    # 输出训练集、验证集和测试集的数量
    train_count = len(train_loader.dataset)
    val_count = len(val_loader.dataset)
    test_count = len(test_loader.dataset)

    print(f"训练集数量: {train_count}")
    print(f"验证集数量: {val_count}")
    print(f"测试集数量: {test_count}")
    
    # 计算并输出每个数据集中包含的图像数量和描述数量
    def count_images_and_descriptions(dataset):
        image_ids = set()
        description_count = 0
        for _, _, img_id in dataset:
            image_ids.add(img_id)
            description_count += 1
        return len(image_ids), description_count

    train_images, train_descriptions = count_images_and_descriptions(train_loader.dataset)
    val_images, val_descriptions = count_images_and_descriptions(val_loader.dataset)
    test_images, test_descriptions = count_images_and_descriptions(test_loader.dataset)

    print(f"训练集包含图像数量: {train_images}, 描述数量: {train_descriptions}")
    print(f"验证集包含图像数量: {val_images}, 描述数量: {val_descriptions}")
    print(f"测试集包含图像数量: {test_images}, 描述数量: {test_descriptions}")

    
    for step, (image,text) in tqdm(enumerate(train_loader), leave=False):
        print(image.shape)
        # print(text)
    # 遍历验证数据
    #加的
    for step, (image, text) in tqdm(enumerate(val_loader), leave=False):
        print(image.shape)
        # print(text)
    for step, (image,text) in tqdm(enumerate(test_loader), leave=False):
        print(image.shape)
        # print(text)
'''

if __name__ == '__main__':
    options = parser_options()     # 解析配置文件
    # 获取数据加载器
    train_loader, val_loader, test_loader = get_loaders(options["dataset"]["batch_size"], options)
    
    # 计算并输出每个数据集中包含的图像数量和描述数量
    def count_images_and_descriptions(dataset):
        image_ids = set()
        description_count = 0
        for _, _, img_id in tqdm(dataset, desc="统计数据", unit="项"):
            image_ids.add(img_id)
            description_count += 1
        return len(image_ids), description_count

    train_images, train_descriptions = count_images_and_descriptions(train_loader.dataset)
    val_images, val_descriptions = count_images_and_descriptions(val_loader.dataset)
    test_images, test_descriptions = count_images_and_descriptions(test_loader.dataset)

    print(f"训练集包含图像数量: {train_images}, 描述数量: {train_descriptions}")
    print(f"验证集包含图像数量: {val_images}, 描述数量: {val_descriptions}")
    print(f"测试集包含图像数量: {test_images}, 描述数量: {test_descriptions}")

    # 遍历训练数据
    for step, (image, text, img_id) in tqdm(enumerate(train_loader), desc="训练数据", unit="批次", leave=False):
        print(image.shape)
        # print(text)
    
    # 遍历验证数据
    for step, (image, text, img_id) in tqdm(enumerate(val_loader), desc="验证数据", unit="批次", leave=False):
        print(image.shape)
        # print(text)
    
    # 遍历测试数据
    for step, (image, text, img_id) in tqdm(enumerate(test_loader), desc="测试数据", unit="批次", leave=False):
        print(image.shape)
        # print(text)
    
    
'''
    # 遍历训练数据
    for step, (image, text, img_id) in tqdm(enumerate(train_loader), leave=False):
        print(image.shape)
        # print(text)
    
    # 遍历验证数据
    for step, (image, text, img_id) in tqdm(enumerate(val_loader), leave=False):
        print(image.shape)
        # print(text)
    
    # 遍历测试数据
    for step, (image, text, img_id) in tqdm(enumerate(test_loader), leave=False):
        print(image.shape)
        # print(text)
'''  


def count_images_and_descriptions(dataset):
    image_ids = set()
    description_count = 0
    category_counts = {}  # 用于统计每个类别的数量

    for _, _, img_id in tqdm(dataset, desc="统计数据", unit="项"):
        image_ids.add(img_id)
        description_count += 1

        # 获取类别信息（假设类别信息存储在 dataset 中）
        category = dataset[img_id]['category']
        if category not in category_counts:
            category_counts[category] = 0
        category_counts[category] += 1

    # 输出每个类别的数据数量
    for category, count in category_counts.items():
        print(f"类别 {category} 的数量: {count}")

    return len(image_ids), description_count