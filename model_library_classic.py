import numpy as np 
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics 
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import torchvision
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter

from PIL import Image
import cv2
import albumentations as A
import imgaug

import time, math, random
import os,gc,psutil,sys
import glob
from tqdm import tqdm

from torchsummary import summary
import segmentation_models_pytorch as smp


def create_df(df_removed_list, df_removed, IMAGE_PATH, datasettype): #if df_removed is True list is not considered
    
    name = []
    for dirname, _, filenames in os.walk(IMAGE_PATH):
        for filename in filenames:
            
            imname = filename.split('.')[0]

            if (datasettype == "synthetic-moon"): imname = imname.replace('render','')
            if (datasettype == "real-moon"): imname = imname
            if (datasettype == "ai4mars"): 
                imname = imname.replace("_merged","")
                if (imname == 'NLB_432655207EDR_F0160148NCAM00394M1'): continue

            if (df_removed):
                if imname in df_removed_list: continue
            else:
                if imname not in df_removed_list: continue       
            name.append(imname)
    
    return pd.DataFrame({'id': name}, index = np.arange(0, len(name)))

# train-test-validation splitter

def datasplitter(datasettype, df_tot = [], df_train = [], df_val = [], df_test = []):

    if (datasettype == "synthetic-moon"):
        X_trainval, X_test = train_test_split(df_tot['id'].values, test_size = 0.1, random_state=42, shuffle = True)
        X_train, X_val = train_test_split(X_trainval, test_size = 0.1114262, random_state=42, shuffle = True)
    
    if (datasettype == "real-moon"):
        X_trainval, X_test = train_test_split(df_tot['id'].values, test_size = 15, random_state=1, shuffle = True)
        X_train, X_val = train_test_split(X_trainval, test_size = 5, random_state=1, shuffle = True)

    if (datasettype == "ai4mars" or datasettype == "ai4mars-inference"):
        X_train, X_val = train_test_split(df_train['id'].values, test_size=0.20, random_state=42)
        X_test = df_test['id'].values

    if (datasettype == "marsdataset3"):
        X_train = df_train['id'].values
        X_val = df_val['id'].values
        X_test = df_test['id'].values

    print('Train Size   : ', len(X_train))
    print('Val Size     : ', len(X_val))
    print('Test Size    : ', len(X_test))

    return X_train, X_val, X_test

def plot_prediction(image,mask,prediction,dataname,bal_acc,idata,npixx,npixy,figsdir):

    col_dict = {0: "red",
                1: "blue",
                2: "gold"}
    # We create a colormar from our list of colors
    cm = matplotlib.colors.ListedColormap([col_dict[x] for x in col_dict.keys()])

    # Let's also define the description of each category : 1 (blue) is Sea; 2 (red) is burnt, etc... Order should be respected here ! Or using another dict maybe could help.
    labels = np.array(["Rocks", "Sky", "Terrain"])
    len_lab = len(labels)

    # prepare normalizer
    # Prepare bins for the normalizer
    norm_bins = np.sort([*col_dict.keys()]) + 0.5
    norm_bins = np.insert(norm_bins, 0, np.min(norm_bins) - 1.0)
    # print(norm_bins)
    # Make normalizer and formatter
    norm = matplotlib.colors.BoundaryNorm(norm_bins, len_lab, clip=True)
    fmt = matplotlib.ticker.FuncFormatter(lambda x, pos: labels[norm(x)])

    # Plot our figure
    fig, ax = plt.subplots()
    ax.imshow(image,cmap="gray")
    im = ax.imshow(prediction, alpha = 0.6, cmap=cm, norm=norm)
    diff = norm_bins[1:] - norm_bins[:-1]
    tickz = norm_bins[:-1] + diff / 2
    cb = fig.colorbar(im, format=fmt, ticks=tickz, orientation='horizontal')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(figsdir + "pred_%i.pdf" % idata)
    plt.close('all')


    fig, ax = plt.subplots()
    ax.imshow(image,cmap="gray")
    im = ax.imshow(mask, alpha = 0.6, cmap=cm, norm=norm)
    diff = norm_bins[1:] - norm_bins[:-1]
    tickz = norm_bins[:-1] + diff / 2
    cb = fig.colorbar(im, format=fmt, ticks=tickz, orientation='horizontal')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(figsdir + "gt_%i.pdf" % idata)
    plt.close('all')
    gc.collect()


def plot_conf_matrix(conf_matrix, figsdir):

    plt.figure()
    sns.heatmap(pd.DataFrame(conf_matrix.tolist(),
                            columns = ["Rocks","Sky","Terrain"],index = ["Rocks","Sky","Terrain"]), 
                            annot = True,
                            vmin = 0.,vmax = 1.)
    plt.ylabel("Ground Truth")
    plt.xlabel("Predicted Labels")
    plt.savefig(figsdir + "conf_matrix.png")

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Modified Balanced Accuracy

def balanced_accuracy_score(y_true, y_pred):
    
    #should use labels = [0,1,2]
    C = metrics.confusion_matrix(y_true, y_pred, labels = [0,1,2])
    
    with np.errstate(divide="ignore", invalid="ignore"):
        per_class = np.diag(C) / C.sum(axis=1) #tp/(tp+fn)
        
    for irec,rec in enumerate(per_class):
        #label is not in ground truth and is not predicted
        if (np.isnan(rec) and C.sum(axis=0)[irec] == 0): rec = 1 #no true pos, no true neg and no false positive
   
    #if there are false positives but no false negative and true positive remains nan
    per_class = per_class[~np.isnan(per_class)]
    
    score = np.nanmean(per_class)
    return score


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Dataset Class

class FakeMoonDataset(Dataset):
    
    def __init__(self, img_path, mask_path, X, transform, algo):
        self.img_path = img_path
        self.mask_path = mask_path
        self.X = X
        self.transform = transform
        self.algo = algo
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        
        #lettura immagini
        img = cv2.imread(self.img_path + 'render' + self.X[idx] + '.png')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #from BGR to RGB
        mask = cv2.imread(self.mask_path + 'ground' + self.X[idx] + '.png')
        
        #merging mask
        mask = self.merge_moon_synth(mask)
        
        #converting to b/w labels
        mask = self.mask_to_label(mask)
        
        #transformations
        aug = self.transform(image=img, mask=mask)
        _ = Image.fromarray(aug['image'])
        mask_aug = aug['mask']

        #algorithm (Otsu/Canny)
        output = self.algorithm(img)
        output = self.mask_to_label(output)

        #transformations
        aug = self.transform(image=img, mask=output)
        img_aug = Image.fromarray(aug['image'])
        output_aug = aug['mask']
            
        return [img, img_aug, output, output_aug, mask, mask_aug]
    
    def algorithm(self, image):

        #otsu
        if (self.algo == "otsu"):

            # Backgorund selection
            ret,th = cv2.threshold(image,0,255,cv2.THRESH_OTSU)
            th = cv2.bitwise_not(cv2.dilate(th,None))

            # Sky mask
            contours, hierarchy = cv2.findContours(th, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            area=[]
            for _,cntr in enumerate(contours):
                area.append(cv2.contourArea(cntr))

            if (len(area) == 0):
                sky_mask = np.zeros_like(image,dtype = np.uint8)
            else: 
                sky_idx = area.index(max(area))
                sky_mask = cv2.drawContours(np.zeros(np.shape(image), np.uint8), contours, sky_idx, 255, -1)

            img_skymasked = cv2.bitwise_and(image, image, mask=cv2.bitwise_not(sky_mask))

            # Terrain and Rocks masks 
            ret,terrain_mask = cv2.threshold(img_skymasked,0,255,cv2.THRESH_OTSU)
            rocks_mask = cv2.bitwise_and(cv2.bitwise_not(terrain_mask),cv2.bitwise_not(terrain_mask), mask=cv2.bitwise_not(sky_mask))

            # Final mask
            mask_dims = np.repeat(image[:, :, np.newaxis], 3, axis=2)
            mask_new = np.zeros_like(mask_dims,dtype = np.uint8)
            mask_new[:,:,0] = np.array(rocks_mask,dtype=np.uint8)*255
            mask_new[:,:,1] = np.array(sky_mask,dtype=np.uint8)*255
            mask_new[:,:,2] = np.array(terrain_mask,dtype=np.uint8)*255

            return mask_new
        
        #canny
        if (self.algo == "canny"):

            terrain_mask = cv2.dilate(cv2.Canny(image,0,255),None)

            # Sky mask
            contours, hierarchy = cv2.findContours(cv2.bitwise_not(terrain_mask), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            area=[]
            for i,cntr in enumerate(contours):
                area.append(cv2.contourArea(cntr))

            if (len(area) == 0):
                sky_mask = np.zeros_like(image,dtype = np.uint8)
            else: 
                sky_idx = area.index(max(area))
                sky_mask = cv2.drawContours(np.zeros(np.shape(image), np.uint8), contours, sky_idx, 255, -1)

            # Rocks mask
            rocks_mask = cv2.bitwise_and(cv2.bitwise_not(terrain_mask),cv2.bitwise_not(terrain_mask), mask=cv2.bitwise_not(sky_mask))

            # Final mask
            mask_dims = np.repeat(image[:, :, np.newaxis], 3, axis=2)
            mask_new = np.zeros_like(mask_dims,dtype = np.uint8)
            mask_new[:,:,0] = np.array(rocks_mask,dtype=np.uint8)*255
            mask_new[:,:,1] = np.array(sky_mask,dtype=np.uint8)*255
            mask_new[:,:,2] = np.array(terrain_mask,dtype=np.uint8)*255

            return mask_new
        
        #classic
        if (self.algo == "hybrid"):

            # Backgorund selection
            ret,th = cv2.threshold(image,0,255,cv2.THRESH_OTSU)
            th = cv2.bitwise_not(cv2.dilate(th,None))

            # Sky mask
            contours, hierarchy = cv2.findContours(th, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            area=[]
            for _,cntr in enumerate(contours):
                area.append(cv2.contourArea(cntr))

            if (len(area) == 0):
                sky_mask = np.zeros_like(image,dtype = np.uint8)
            else: 
                sky_idx = area.index(max(area))
                sky_mask = cv2.drawContours(np.zeros(np.shape(image), np.uint8), contours, sky_idx, 255, -1)

            img_skymasked = cv2.bitwise_and(image, image, mask=cv2.bitwise_not(sky_mask))

            # Terrain and Rocks masks
            rocks_mask = cv2.dilate(cv2.Canny(img_skymasked,0,255,apertureSize=3),None)
            terrain_mask = cv2.bitwise_not(cv2.bitwise_or(rocks_mask,sky_mask))

            # Final mask
            mask_dims = np.repeat(image[:, :, np.newaxis], 3, axis=2)
            mask_new = np.zeros_like(mask_dims,dtype = np.uint8)
            mask_new[:,:,0] = np.array(rocks_mask,dtype=np.uint8)*255
            mask_new[:,:,1] = np.array(sky_mask,dtype=np.uint8)*255
            mask_new[:,:,2] = np.array(terrain_mask,dtype=np.uint8)*255

            return mask_new

            

    #colors to label in 1d image
    def mask_to_label(self,mask):
        new_mask = np.zeros_like(mask[:,:,0])
        for col in range(mask.shape[2]):
            labmask = np.where(mask[:,:,col]>0,col,0)
            new_mask += labmask.astype(np.uint8)
            
        new_mask = np.where(new_mask==4,3,new_mask)
        new_mask = np.where(new_mask==5,3,new_mask)
        new_mask = np.where(new_mask==6,3,new_mask)
        
        return new_mask
    
    def merge_moon_synth(self,data):
    
        data[data > 127] = 255
        data[data <= 127] = 0

        # 0 = big rocks
        # 1 = smaller rocks
        # 2 = sky
        data_bigrocks = data[:,:,0]
        data_smallrocks = data[:,:,1]
        data_sky = data[:,:,2]

        #merging rocks
        rocks = np.logical_or(data[:,:,0],data[:,:,1]) 
        rocks = np.array(rocks, dtype = np.uint8)*255

        #black skies in superposition
        blacksky_lines = np.logical_and(rocks, data_sky) #horizon creates a superposition
        blacksky_lines = np.array(blacksky_lines, dtype = np.uint8)*255
        data_sky[blacksky_lines != 0] = 0
        rocks[blacksky_lines != 0] = 255

        # creating new class: terrain
        data_terrain = np.logical_and(data_sky == 0, rocks == 0)
        np.shape(data_terrain)
        data_terrain = np.array(data_terrain,dtype=np.uint8)*255

        ##### new mask dataset
        data_new = np.zeros_like(data,dtype = np.uint8)
        data_new[:,:,0] = rocks
        data_new[:,:,1] = data_sky
        data_new[:,:,2] = data_terrain
        
        return data_new
    
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------


class RealMoonDataset(Dataset):
    
    def __init__(self, img_path, mask_path, X, transform, algo):
        self.img_path = img_path
        self.mask_path = mask_path
        self.X = X
        self.transform = transform
        self.algo = algo
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        
        #lettura immagini
        img = cv2.imread(self.img_path + self.X[idx] + '.png')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #from BGR to RGB
        mask = cv2.imread(self.mask_path + 'g_' + self.X[idx] + '.png')
        
        #merging mask
        mask = self.merge_moon_synth(mask)
        
        #converting to b/w labels
        mask = self.mask_to_label(mask)
        
        #transformations
        aug = self.transform(image=img, mask=mask)
        _ = Image.fromarray(aug['image'])
        mask_aug = aug['mask']

        #algorithm (Otsu/Canny)
        output = self.algorithm(img)
        output = self.mask_to_label(output)

        #transformations
        aug = self.transform(image=img, mask=output)
        img_aug = Image.fromarray(aug['image'])
        output_aug = aug['mask']
            
        return [img, img_aug, output, output_aug, mask, mask_aug]
    
    #algorithm
    def algorithm(self, image):

        #otsu
        if (self.algo == "otsu"):

            # Backgorund selection
            ret,th = cv2.threshold(image,0,255,cv2.THRESH_OTSU)
            th = cv2.bitwise_not(cv2.dilate(th,None))

            # Sky mask
            contours, hierarchy = cv2.findContours(th, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            area=[]
            for _,cntr in enumerate(contours):
                area.append(cv2.contourArea(cntr))

            if (len(area) == 0):
                sky_mask = np.zeros_like(image,dtype = np.uint8)
            else: 
                sky_idx = area.index(max(area))
                sky_mask = cv2.drawContours(np.zeros(np.shape(image), np.uint8), contours, sky_idx, 255, -1)

            img_skymasked = cv2.bitwise_and(image, image, mask=cv2.bitwise_not(sky_mask))

            # Terrain and Rocks masks 
            ret,terrain_mask = cv2.threshold(img_skymasked,0,255,cv2.THRESH_OTSU)
            rocks_mask = cv2.bitwise_and(cv2.bitwise_not(terrain_mask),cv2.bitwise_not(terrain_mask), mask=cv2.bitwise_not(sky_mask))

            # Final mask
            mask_dims = np.repeat(image[:, :, np.newaxis], 3, axis=2)
            mask_new = np.zeros_like(mask_dims,dtype = np.uint8)
            mask_new[:,:,0] = np.array(rocks_mask,dtype=np.uint8)*255
            mask_new[:,:,1] = np.array(sky_mask,dtype=np.uint8)*255
            mask_new[:,:,2] = np.array(terrain_mask,dtype=np.uint8)*255

            return mask_new
        
        #canny
        if (self.algo == "canny"):

            terrain_mask = cv2.dilate(cv2.Canny(image,0,255),None)

            # Sky mask
            contours, hierarchy = cv2.findContours(cv2.bitwise_not(terrain_mask), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            area=[]
            for i,cntr in enumerate(contours):
                area.append(cv2.contourArea(cntr))

            if (len(area) == 0):
                sky_mask = np.zeros_like(image,dtype = np.uint8)
            else: 
                sky_idx = area.index(max(area))
                sky_mask = cv2.drawContours(np.zeros(np.shape(image), np.uint8), contours, sky_idx, 255, -1)

            # Rocks mask
            rocks_mask = cv2.bitwise_and(cv2.bitwise_not(terrain_mask),cv2.bitwise_not(terrain_mask), mask=cv2.bitwise_not(sky_mask))

            # Final mask
            mask_dims = np.repeat(image[:, :, np.newaxis], 3, axis=2)
            mask_new = np.zeros_like(mask_dims,dtype = np.uint8)
            mask_new[:,:,0] = np.array(rocks_mask,dtype=np.uint8)*255
            mask_new[:,:,1] = np.array(sky_mask,dtype=np.uint8)*255
            mask_new[:,:,2] = np.array(terrain_mask,dtype=np.uint8)*255

            return mask_new
        
        #classic
        if (self.algo == "hybrid"):

            # Backgorund selection
            ret,th = cv2.threshold(image,0,255,cv2.THRESH_OTSU)
            th = cv2.bitwise_not(cv2.dilate(th,None))

            # Sky mask
            contours, hierarchy = cv2.findContours(th, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            area=[]
            for _,cntr in enumerate(contours):
                area.append(cv2.contourArea(cntr))
    
            if (len(area) == 0):
                sky_mask = np.zeros_like(image,dtype = np.uint8)
            else: 
                sky_idx = area.index(max(area))
                sky_mask = cv2.drawContours(np.zeros(np.shape(image), np.uint8), contours, sky_idx, 255, -1)

            img_skymasked = cv2.bitwise_and(image, image, mask=cv2.bitwise_not(sky_mask))

            # Terrain and Rocks masks
            rocks_mask = cv2.dilate(cv2.Canny(img_skymasked,0,255,apertureSize=3),None)
            terrain_mask = cv2.bitwise_not(cv2.bitwise_or(rocks_mask,sky_mask))

            # Final mask
            mask_dims = np.repeat(image[:, :, np.newaxis], 3, axis=2)
            mask_new = np.zeros_like(mask_dims,dtype = np.uint8)
            mask_new[:,:,0] = np.array(rocks_mask,dtype=np.uint8)*255
            mask_new[:,:,1] = np.array(sky_mask,dtype=np.uint8)*255
            mask_new[:,:,2] = np.array(terrain_mask,dtype=np.uint8)*255

            return mask_new
    
    #colors to label in 1d image
    def mask_to_label(self,mask):
        new_mask = np.zeros_like(mask[:,:,0])
        for col in range(mask.shape[2]):
            labmask = np.where(mask[:,:,col]>0,col,0)
            new_mask += labmask.astype(np.uint8)
            
        new_mask = np.where(new_mask==4,3,new_mask)
        new_mask = np.where(new_mask==5,3,new_mask)
        new_mask = np.where(new_mask==6,3,new_mask)
        
        return new_mask
    
    def merge_moon_synth(self,data):
    
        data[data > 127] = 255
        data[data <= 127] = 0

        # 0 = big rocks
        # 1 = smaller rocks
        # 2 = sky
        data_bigrocks = data[:,:,0]
        data_smallrocks = data[:,:,1]
        data_sky = data[:,:,2]

        #merging rocks
        rocks = np.logical_or(data[:,:,0],data[:,:,1]) 
        rocks = np.array(rocks, dtype = np.uint8)*255

        #black skies in superposition
        blacksky_lines = np.logical_and(rocks, data_sky) #horizon creates a superposition
        blacksky_lines = np.array(blacksky_lines, dtype = np.uint8)*255
        data_sky[blacksky_lines != 0] = 0
        rocks[blacksky_lines != 0] = 255

        # creating new class: terrain
        data_terrain = np.logical_and(data_sky == 0, rocks == 0)
        np.shape(data_terrain)
        data_terrain = np.array(data_terrain,dtype=np.uint8)*255

        ##### new mask dataset
        data_new = np.zeros_like(data,dtype = np.uint8)
        data_new[:,:,0] = rocks
        data_new[:,:,1] = data_sky
        data_new[:,:,2] = data_terrain
        
        return data_new
    
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------

class AI4MarsDataset(Dataset):
    
    def __init__(self, img_path, mask_path, rng_path, X, transform, algo, test = False):
        self.img_path = img_path
        self.mask_path = mask_path
        self.range_path = rng_path
        self.X = X
        self.transform = transform
        self.test = test
        self.algo = algo
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        
        #lettura immagini
        img = cv2.imread(self.img_path + self.X[idx] + '.JPG')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #from BGR to RGB
        if (self.test): mask = cv2.imread(self.mask_path + self.X[idx] + '_merged.png')
        else: mask = cv2.imread(self.mask_path + self.X[idx] + '.png')
        rng30 = cv2.imread(self.range_path + (self.X[idx]).replace("EDR","RNG") + '.png')
        
        #merging mask
        mask = self.merge_real_mars(mask,rng30)
        
        #transformations
        aug = self.transform(image=img, mask=mask)
        _ = Image.fromarray(aug['image'])
        mask_aug = aug['mask']

        #algorithm (Otsu/Canny)
        output = self.algorithm(img)
        output = self.mask_to_label(output)

        #transformations
        aug = self.transform(image=img, mask=output)
        img_aug = Image.fromarray(aug['image'])
        output_aug = aug['mask']
            
        return [img, img_aug, output, output_aug, mask, mask_aug]
        
    
    def merge_real_mars(self,mask,rng):

        ground_truth = np.zeros(mask[:,:,0].shape)
        ground_truth[:,:] = mask[:,:,0]

        range_30m = np.zeros((rng[:,:,0].shape))
        range_30m[:,:] = rng[:,:,0]

        #adding null and sky
        ground_truth[ground_truth == 255] = 4
        ground_truth[range_30m == 1] = 5

        #all components
        ground_truth_0 = np.where(ground_truth == 0, 1, 0)
        ground_truth_1 = np.where(ground_truth == 1, 1, 0)
        ground_truth_2 = np.where(ground_truth == 2, 1, 0)
        ground_truth_3 = np.where(ground_truth == 3, 1, 0)
        ground_truth_4 = np.where(ground_truth == 4, 1, 0)
        ground_truth_5 = np.where(range_30m == 1, 1, 0)

        ground_truth_rocks = ground_truth_3
        ground_truth_terrain = np.logical_or(ground_truth_0,ground_truth_1)
        ground_truth_terrain = np.logical_or(ground_truth_terrain,ground_truth_2)
        ground_truth_terrain = np.logical_or(ground_truth_terrain,ground_truth_4)
        ground_truth_sky = ground_truth_5

        #new mask
        merged_mask = np.zeros((1024,1024))
        merged_mask[ground_truth_rocks == 1] = 0
        merged_mask[ground_truth_sky == 1] = 1
        merged_mask[ground_truth_terrain] = 2
        
        return merged_mask
    
    #algorithm
    def algorithm(self, image):

        #otsu
        if (self.algo == "otsu"):

            # Backgorund selection
            ret,th = cv2.threshold(image,0,255,cv2.THRESH_OTSU)
            th = cv2.bitwise_not(cv2.dilate(th,None))

            # Sky mask
            contours, hierarchy = cv2.findContours(th, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            area=[]
            for _,cntr in enumerate(contours):
                area.append(cv2.contourArea(cntr))

            if (len(area) == 0):
                sky_mask = np.zeros_like(image,dtype = np.uint8)
            else: 
                sky_idx = area.index(max(area))
                sky_mask = cv2.drawContours(np.zeros(np.shape(image), np.uint8), contours, sky_idx, 255, -1)

            img_skymasked = cv2.bitwise_and(image, image, mask=cv2.bitwise_not(sky_mask))

            # Terrain and Rocks masks 
            ret,terrain_mask = cv2.threshold(img_skymasked,0,255,cv2.THRESH_OTSU)
            rocks_mask = cv2.bitwise_and(cv2.bitwise_not(terrain_mask),cv2.bitwise_not(terrain_mask), mask=cv2.bitwise_not(sky_mask))

            # Final mask
            mask_dims = np.repeat(image[:, :, np.newaxis], 3, axis=2)
            mask_new = np.zeros_like(mask_dims,dtype = np.uint8)
            mask_new[:,:,0] = np.array(rocks_mask,dtype=np.uint8)*255
            mask_new[:,:,1] = np.array(sky_mask,dtype=np.uint8)*255
            mask_new[:,:,2] = np.array(terrain_mask,dtype=np.uint8)*255

            return mask_new
        
        #canny
        if (self.algo == "canny"):

            terrain_mask = cv2.dilate(cv2.Canny(image,0,255),None)

            # Sky mask
            contours, hierarchy = cv2.findContours(cv2.bitwise_not(terrain_mask), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            area=[]
            for i,cntr in enumerate(contours):
                area.append(cv2.contourArea(cntr))

            if (len(area) == 0):
                sky_mask = np.zeros_like(image,dtype = np.uint8)
            else: 
                sky_idx = area.index(max(area))
                sky_mask = cv2.drawContours(np.zeros(np.shape(image), np.uint8), contours, sky_idx, 255, -1)

            # Rocks mask
            rocks_mask = cv2.bitwise_and(cv2.bitwise_not(terrain_mask),cv2.bitwise_not(terrain_mask), mask=cv2.bitwise_not(sky_mask))

            # Final mask
            mask_dims = np.repeat(image[:, :, np.newaxis], 3, axis=2)
            mask_new = np.zeros_like(mask_dims,dtype = np.uint8)
            mask_new[:,:,0] = np.array(rocks_mask,dtype=np.uint8)*255
            mask_new[:,:,1] = np.array(sky_mask,dtype=np.uint8)*255
            mask_new[:,:,2] = np.array(terrain_mask,dtype=np.uint8)*255

            return mask_new
        
        #classic
        if (self.algo == "hybrid"):

            # Backgorund selection
            ret,th = cv2.threshold(image,0,255,cv2.THRESH_OTSU)
            th = cv2.bitwise_not(cv2.dilate(th,None))

            # Sky mask
            contours, hierarchy = cv2.findContours(th, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            area=[]
            for _,cntr in enumerate(contours):
                area.append(cv2.contourArea(cntr))
    
            if (len(area) == 0):
                sky_mask = np.zeros_like(image,dtype = np.uint8)
            else: 
                sky_idx = area.index(max(area))
                sky_mask = cv2.drawContours(np.zeros(np.shape(image), np.uint8), contours, sky_idx, 255, -1)

            img_skymasked = cv2.bitwise_and(image, image, mask=cv2.bitwise_not(sky_mask))

            # Terrain and Rocks masks
            rocks_mask = cv2.dilate(cv2.Canny(img_skymasked,0,255,apertureSize=3),None)
            terrain_mask = cv2.bitwise_not(cv2.bitwise_or(rocks_mask,sky_mask))

            # Final mask
            mask_dims = np.repeat(image[:, :, np.newaxis], 3, axis=2)
            mask_new = np.zeros_like(mask_dims,dtype = np.uint8)
            mask_new[:,:,0] = np.array(rocks_mask,dtype=np.uint8)*255
            mask_new[:,:,1] = np.array(sky_mask,dtype=np.uint8)*255
            mask_new[:,:,2] = np.array(terrain_mask,dtype=np.uint8)*255

            return mask_new
    
    #colors to label in 1d image
    def mask_to_label(self,mask):
        new_mask = np.zeros_like(mask[:,:,0])
        for col in range(mask.shape[2]):
            labmask = np.where(mask[:,:,col]>0,col,0)
            new_mask += labmask.astype(np.uint8)
            
        new_mask = np.where(new_mask==4,3,new_mask)
        new_mask = np.where(new_mask==5,3,new_mask)
        new_mask = np.where(new_mask==6,3,new_mask)
        
        return new_mask

# MarsDataset-v3
class MarsDatasetv3(Dataset):
    
    def __init__(self, img_path, mask_path, sky_path, X, transform, algo):
        self.img_path = img_path
        self.mask_path = mask_path
        self.sky_path = sky_path
        self.X = X
        self.transform = transform
        self.algo = algo
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        
        #lettura immagini
        img = cv2.imread(self.img_path + self.X[idx] + '.png')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #from BGR to RGB
        mask = cv2.imread(self.mask_path + self.X[idx] + '.png')

        if os.path.isfile(self.sky_path + self.X[idx] + '.png'): 
            mask_sky = np.array(cv2.imread(self.sky_path + self.X[idx] + '.png'))
            mask_sky[mask_sky > 127] = 255
            mask_sky[mask_sky <= 127] = 0
            data_sky = mask_sky[:,:,2] 
        else:
            mask_sky = np.zeros_like(mask)
            data_sky = mask_sky[:,:,2]

        #merging mask
        mask = self.merge_moon_mars(mask,data_sky)
        
        #transformations
        aug = self.transform(image=img, mask=mask)
        _ = Image.fromarray(aug['image'])
        mask_aug = aug['mask']

        #algorithm (Otsu/Canny)
        output = self.algorithm(img)
        output = self.mask_to_label(output)

        #transformations
        aug = self.transform(image=img, mask=output)
        img_aug = Image.fromarray(aug['image'])
        output_aug = aug['mask']
            
        return [img, img_aug, output, output_aug, mask, mask_aug]
            
         
    
   #algorithm
    def algorithm(self, image):

        #otsu
        if (self.algo == "otsu"):

            # Backgorund selection
            ret,th = cv2.threshold(image,0,255,cv2.THRESH_OTSU)
            th = cv2.bitwise_not(cv2.dilate(th,None))

            # Sky mask
            contours, hierarchy = cv2.findContours(th, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            area=[]
            for _,cntr in enumerate(contours):
                area.append(cv2.contourArea(cntr))

            if (len(area) == 0):
                sky_mask = np.zeros_like(image,dtype = np.uint8)
            else: 
                sky_idx = area.index(max(area))
                sky_mask = cv2.drawContours(np.zeros(np.shape(image), np.uint8), contours, sky_idx, 255, -1)

            img_skymasked = cv2.bitwise_and(image, image, mask=cv2.bitwise_not(sky_mask))

            # Terrain and Rocks masks 
            ret,terrain_mask = cv2.threshold(img_skymasked,0,255,cv2.THRESH_OTSU)
            rocks_mask = cv2.bitwise_and(cv2.bitwise_not(terrain_mask),cv2.bitwise_not(terrain_mask), mask=cv2.bitwise_not(sky_mask))

            # Final mask
            mask_dims = np.repeat(image[:, :, np.newaxis], 3, axis=2)
            mask_new = np.zeros_like(mask_dims,dtype = np.uint8)
            mask_new[:,:,0] = np.array(rocks_mask,dtype=np.uint8)*255
            mask_new[:,:,1] = np.array(sky_mask,dtype=np.uint8)*255
            mask_new[:,:,2] = np.array(terrain_mask,dtype=np.uint8)*255

            return mask_new
        
        #canny
        if (self.algo == "canny"):

            terrain_mask = cv2.dilate(cv2.Canny(image,0,255),None)

            # Sky mask
            contours, hierarchy = cv2.findContours(cv2.bitwise_not(terrain_mask), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            area=[]
            for i,cntr in enumerate(contours):
                area.append(cv2.contourArea(cntr))

            if (len(area) == 0):
                sky_mask = np.zeros_like(image,dtype = np.uint8)
            else: 
                sky_idx = area.index(max(area))
                sky_mask = cv2.drawContours(np.zeros(np.shape(image), np.uint8), contours, sky_idx, 255, -1)

            # Rocks mask
            rocks_mask = cv2.bitwise_and(cv2.bitwise_not(terrain_mask),cv2.bitwise_not(terrain_mask), mask=cv2.bitwise_not(sky_mask))

            # Final mask
            mask_dims = np.repeat(image[:, :, np.newaxis], 3, axis=2)
            mask_new = np.zeros_like(mask_dims,dtype = np.uint8)
            mask_new[:,:,0] = np.array(rocks_mask,dtype=np.uint8)*255
            mask_new[:,:,1] = np.array(sky_mask,dtype=np.uint8)*255
            mask_new[:,:,2] = np.array(terrain_mask,dtype=np.uint8)*255

            return mask_new
        
        #classic
        if (self.algo == "hybrid"):

            # Backgorund selection
            ret,th = cv2.threshold(image,0,255,cv2.THRESH_OTSU)
            th = cv2.bitwise_not(cv2.dilate(th,None))

            # Sky mask
            contours, hierarchy = cv2.findContours(th, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            area=[]
            for _,cntr in enumerate(contours):
                area.append(cv2.contourArea(cntr))
    
            if (len(area) == 0):
                sky_mask = np.zeros_like(image,dtype = np.uint8)
            else: 
                sky_idx = area.index(max(area))
                sky_mask = cv2.drawContours(np.zeros(np.shape(image), np.uint8), contours, sky_idx, 255, -1)

            img_skymasked = cv2.bitwise_and(image, image, mask=cv2.bitwise_not(sky_mask))

            # Terrain and Rocks masks
            rocks_mask = cv2.dilate(cv2.Canny(img_skymasked,0,255,apertureSize=3),None)
            terrain_mask = cv2.bitwise_not(cv2.bitwise_or(rocks_mask,sky_mask))

            # Final mask
            mask_dims = np.repeat(image[:, :, np.newaxis], 3, axis=2)
            mask_new = np.zeros_like(mask_dims,dtype = np.uint8)
            mask_new[:,:,0] = np.array(rocks_mask,dtype=np.uint8)*255
            mask_new[:,:,1] = np.array(sky_mask,dtype=np.uint8)*255
            mask_new[:,:,2] = np.array(terrain_mask,dtype=np.uint8)*255

            return mask_new
        
    
    def merge_moon_mars(self,data,data_sky):

        # if present, merge the sky mask with rocks mask (class #1 = sky)
        # if not, change just the classes number into:
        # 0 = rocks
        # 2 = terrain

        data[data > 127] = 255
        data[data <= 127] = 0

        data_rocks = data[:,:,2]
        data_terrain = np.zeros_like(data_rocks)
        data_terrain[data_rocks != 255] = 255

        #black skies in superposition (sky - terrain)
        blacksky_lines = np.logical_and(data_terrain, data_sky) #horizon creates a superposition
        blacksky_lines = np.array(blacksky_lines, dtype = np.uint8)*255
        data_sky[blacksky_lines != 0] = 255
        data_terrain[blacksky_lines != 0] = 0

        #black skies in superposition (sky - rocks)
        blacksky_lines = np.logical_and(data_rocks, data_sky) #horizon creates a superposition
        blacksky_lines = np.array(blacksky_lines, dtype = np.uint8)*255
        data_sky[blacksky_lines != 0] = 0
        data_rocks[blacksky_lines != 0] = 255

        #new mask
        data_new = np.zeros_like(data_sky)
        data_new[data_rocks != 0] = 0
        data_new[data_sky != 0] = 1
        data_new[data_terrain != 0] = 2
        

        return data_new
    
    #colors to label in 1d image
    def mask_to_label(self,mask):
        new_mask = np.zeros_like(mask[:,:,0])
        for col in range(mask.shape[2]):
            labmask = np.where(mask[:,:,col]>0,col,0)
            new_mask += labmask.astype(np.uint8)
            
        new_mask = np.where(new_mask==4,3,new_mask)
        new_mask = np.where(new_mask==5,3,new_mask)
        new_mask = np.where(new_mask==6,3,new_mask)
        
        return new_mask
    
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
