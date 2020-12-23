import csv
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch
from os import listdir
from os.path import isfile, join
import cv2

def crop_image_from_gray(img,tol=7):
    if img.ndim ==2:
        mask = img>tol
        return img[np.ix_(mask.any(1),mask.any(0))]
    elif img.ndim==3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img>tol
        
        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        if (check_shape == 0): # image is too dark so that we crop out everything,
            return img # return original image
        else:
            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
    #         print(img1.shape,img2.shape,img3.shape)
            img = np.stack([img1,img2,img3],axis=-1)
    #         print(img.shape)
        return img

def load_img(path, sigmaX=10):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = crop_image_from_gray(image)
    image = cv2.resize(image, (220, 220))
    image=cv2.addWeighted ( image,4, cv2.GaussianBlur( image , (0,0) , sigmaX) ,-4 ,128)
    image = Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))     
    return image

class Load_traindata(Dataset):
    def __init__(self, transform=None, target_transform=None, loader=load_img, valid=False, valid_len=5000, use_pseudo=False):
        # initial variable
        imgs = []
        self.valid_imgs = []
        self.train_imgs = []
        label_cnt = np.zeros(5)
        # open training_label.csv
        with open('./data/train.csv') as csvfile:
            rows = csv.reader(csvfile)
            for row in rows:
                if row[0] == 'id_code':
                    continue
                label = int(row[1])
                imgs.append((row[0]+'.png', int(row[1]), False))
                label_cnt[int(row[1])] += 1
        
        with open('./data/trainLabels15.csv') as csvfile:
            rows = csv.reader(csvfile)
            for row in rows:
                if row[0] == 'image':
                    continue
                label = int(row[1])
                imgs.append((row[0]+'.jpg', int(row[1]), False))
                label_cnt[int(row[1])] += 1
        
        with open('./data/testLabels15.csv') as csvfile:
            rows = csv.reader(csvfile)
            for row in rows:
                if row[0] == 'image':
                    continue
                label = int(row[1])
                imgs.append((row[0]+'.jpg', int(row[1]), False))
                label_cnt[int(row[1])] += 1
        print(label_cnt)
        # divide data into training set and validation set
        train_len = len(imgs) - valid_len
        self.train_imgs = imgs[0:train_len]
        self.valid_imgs = imgs[-(valid_len+1):-1]

        ccnt = np.zeros(5)
        for i in range(len(self.valid_imgs)):
            ccnt[self.valid_imgs[i][1]] += 1
        print(ccnt)
        if use_pseudo == True:
            with open('./data/pseudo.csv') as csvfile:
                rows = csv.reader(csvfile)
                for row in rows:
                    if row[0] == 'id_code':
                        continue
                    self.train_imgs.append((row[0], int(row[1]), True))
        self.transform = transform
        self.target_transform = target_transform
        self.loader = load_img
        self.valid = valid

    def __getitem__(self, index):
        if self.valid == False:
            filename, label, isPseudo = self.train_imgs[index]
        else:
            filename, label, isPseudo = self.valid_imgs[index]

        if isPseudo == False:
            img = self.loader('./data/train_images/'+filename)
        elif isPseudo == True:
            img = self.loader('./data/extra_images/'+filename)
        
        if self.transform is not None:
            img = self.transform(img)
    
        return img, label

    def __len__(self):
        if self.valid == False:
            return len(self.train_imgs)
        else:
            return len(self.valid_imgs)

class Load_testdata(Dataset):
    def __init__(self, transform=None, target_transform=None, loader=load_img):
        # read the data in testing_data folder
        self.imgs = []
        with open('./data/test.csv') as csvfile:
            rows = csv.reader(csvfile)
            for row in rows:
                if row[0] == 'id_code':
                    continue
                self.imgs.append(row[0])
        self.transform = transform
        self.target_transform = target_transform
        self.loader = load_img

    def __getitem__(self, index):
        filename = self.imgs[index]
        img = self.loader('./data/test_images/'+filename+'.png')
        if self.transform is not None:
            img = self.transform(img)
        return img, filename

    def __len__(self):
        return len(self.imgs)

class Load_extradata(Dataset):
    def __init__(self, transform=None, target_transform=None, loader=load_img):
        # read the data in testing_data folder
        self.imgs = [f for f in listdir('./data/extra_images/') if isfile(join('./data/extra_images/', f))]
        self.transform = transform
        self.target_transform = target_transform
        self.loader = load_img

    def __getitem__(self, index):
        filename = self.imgs[index]
        img = self.loader('./data/extra_images/'+filename)
        if self.transform is not None:
            img = self.transform(img)
        return img, filename

    def __len__(self):
        return len(self.imgs)

# img = load_img("./data/train_images/000c1434d8d7.png")
# img.show()