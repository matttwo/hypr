import torch
import torch.utils.data as data
from PIL import Image
import numpy as np
import os
import sys
from . import transforms 
from collections import defaultdict

from sklearn import preprocessing

def default_loader(path):
    return Image.open(path).convert('RGB')

def Generate_transform_Dict(origin_width=224, width=224, ratio=0.16):
    
    std_value = 1.0 / 255.0
    normalize = transforms.Normalize(mean=[104 / 255.0, 117 / 255.0, 128 / 255.0],
                                     std= [1.0/255, 1.0/255, 1.0/255])

    transform_dict = {}

    transform_dict['rand-crop'] = \
    transforms.Compose([
                transforms.ConvertBGR(),
                transforms.Resize((origin_width, origin_width)),
                transforms.RandomResizedCrop(scale=(ratio, 1), size=width),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
               ])

    transform_dict['center-crop'] = \
    transforms.Compose([
                    transforms.ConvertBGR(),
                    transforms.Resize((origin_width, origin_width)),
                    transforms.CenterCrop(width),
                    transforms.ToTensor(),
                    normalize,
                ])
    
    transform_dict['resize_normalize'] = \
    transforms.Compose([
                    transforms.ConvertBGR(),
                    transforms.Resize((width, width)),
                    transforms.ToTensor(),
                    normalize,
                ])

    transform_dict['resize'] = \
    transforms.Compose([
                    transforms.Resize((width, width)),
                    transforms.ToTensor(),
                ])
    return transform_dict

class MyData(data.Dataset):
    def __init__(self, root=None, root_c=None, label_txt=None,
                 transform=None, loader=default_loader):

        # Initialization data path and train(gallery or query) txt path
        self.root = root
        if root is None:
            self.root = "dataset/"
        
        if label_txt is None:
            label_txt = os.path.join(self.root, 'labels/inat_labels.txt')

        self.transform = transform
        if transform is None:
            self.transform = Generate_transform_Dict()['resize']
        
        file = open(label_txt)
        image_locations_labels = file.readlines()

        images = []
        original_labels = []

        for img_string in image_locations_labels:
            [filename, species] = img_string.split(' ', 1)
            images.append(filename)
            original_labels.append(species)

        le = preprocessing.LabelEncoder()
        le.fit(original_labels)
        labels = le.transform(original_labels)

        classes = list(set(labels))

        # Generate Index Dictionary for every class
        Index = defaultdict(list)
        for i, label in enumerate(labels):
            Index[label].append(i)

        # Initialization Done
        self.images = images
        self.labels = labels
        self.classes = classes
        self.original_labels = original_labels
        self.le = le
        self.Index = Index
        self.loader = loader

    def __getitem__(self, index):
        fn, label = self.images[index], self.labels[index]
        fn = os.path.join(self.root, fn)
        img = self.loader(fn)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.images)

class INat_Birds:
    def __init__(self, root=None, root_c=None, origin_width=224, width=224, ratio=0.16, transform=None, train_share=0.85):
        if transform is None:
            transform = Generate_transform_Dict(origin_width=origin_width, width=width, ratio=ratio)['rand-crop']

        labels_txt = os.path.join(root, 'labels/inat_labels.txt')
        print('Notification: Using {} now!'.format(labels_txt))

        self.dataset = MyData(root, label_txt=labels_txt, transform=transform)

        train_length = int(train_share * self.dataset.__len__())
        test_length = self.dataset.__len__() - train_length
        self.train_data, self.test_data = torch.utils.data.random_split(self.dataset, [train_length, test_length], generator=torch.Generator().manual_seed(42))
            
def test_INat_Birds():
    print(INat_Birds.__name__)
    data = INat_Birds(root='')
    print(len(data.train_data))
    print(len(data.test_data))
    print(data.train_data[1])
    print(data.test_data[1])

if __name__ == "__main__":
    test_INat_Birds()