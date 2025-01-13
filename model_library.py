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
import os,gc,psutil,sys,copy
import glob
from tqdm import tqdm

from torchsummary import summary
import segmentation_models_pytorch as smp

import fvcore.nn
from fvcore.nn import FlopCountAnalysis

import wandb
import unet_custom
from unet_custom import *


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------

#manual memory computation
def computememory():
	mem_threshold = 250*1024*1024 #250 MByte threshold
	mem = psutil.virtual_memory()  # byte units
	# if ((100-mem.percent)<25): print("Memory [Byte] = %i (%.2f %s Total)" % (mem.available,100-mem.percent,"%")) 
	print("Memory available [Byte] = %i (%.2f %s Total)" % (mem.available,100-mem.percent,"%")) 
	if (mem.available <= mem_threshold): 
		print("Memory < Threshold")
		sys.exit()
	return mem.used

#used memory
def used_memory():
    if torch.cuda.is_available():
        print("cuda")
        mem = torch.cuda.mem_get_info()
        return mem
    else:
        mem = psutil.virtual_memory()
        mem_used = mem.total - mem.available
        return mem_used/(1024*1024)

#fixing random seed
def set_seed(seed,device):
    random_seed = seed
    torch.manual_seed(random_seed)
    if (device != "cpu"):
        torch.cuda.manual_seed(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    random.seed(random_seed)
    np.random.seed(random_seed) #if you use numpy
    imgaug.random.seed(seed)
    torch.use_deterministic_algorithms(True,warn_only=True)
    torch.backends.cudnn.benchmark = False
    return 

#string of run name from hyperpars
def run_builder(aug_GaussianBlur, aug_ColorJitter, aug_HorizontalFlip, aug_Rotate, aug_RandomCrop, 
                encoder, optimizer, loss_function,
                baseline,ftuneenc,ftunedec,ftunebnorm,ftuneadapt):

    str_aug=["GaussianBlur","ColorJitter","HorizontalFlip","Rotate","RandomCrop"]
    str_aug=np.array(str_aug)

    index_aug=[aug_GaussianBlur,aug_ColorJitter,aug_HorizontalFlip,aug_Rotate,aug_RandomCrop]
    index_aug=np.array(index_aug)

    index_aug = np.where(index_aug==0)
    index_aug = index_aug[0].tolist()
    str_aug = np.delete(str_aug,index_aug)
    str_aug=str_aug.tolist()

    name_aug = ""

    for name in str_aug:
        name_aug += "_" + name

    name_ftune = ""

    if (baseline == False): 

        str_ftune = ["encoder","decoder","bnorm","adapter"]
        str_ftune = np.array(str_ftune)

        index_ftune = [ftuneenc,ftunedec,ftunebnorm,ftuneadapt]
        index_ftune = np.array(index_ftune)

        index_ftune = np.where(index_ftune==False)
        index_ftune = index_ftune[0].tolist()
        str_ftune = np.delete(str_ftune,index_ftune)
        str_ftune = str_ftune.tolist()

        for name in str_ftune:
            name_ftune += "_" + name
            
    run_name = encoder + "_" + optimizer + "_" + loss_function + name_aug + name_ftune
    if not(baseline or ftuneenc or ftunedec or ftunebnorm or ftuneadapt): 
        run_name = encoder + "_" + optimizer + "_" + loss_function + name_aug
    
    return run_name

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Functions and Classes
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------

def saveorload(modelsdir,color_channels,
               aug_GaussianBlur, aug_ColorJitter, aug_HorizontalFlip, aug_Rotate, aug_RandomCrop, 
               encoder, optimizer, loss_function,
               baseline,ftuneenc,ftunedec,ftunebnorm,ftuneadapt):
    
    if (encoder == "resnet18_relu" or encoder == "resnet18_conv" or encoder == "resnet18_rep2"): encoder_name = "resnet18"
    else: encoder_name = encoder

    if (baseline):
        if os.path.isfile(modelsdir + 'Unet_' + encoder_name + '.pt'): 
            unet = torch.load(modelsdir + 'Unet_' + encoder_name + '.pt')
        else:    
            unet = smp.Unet(encoder_name = encoder, classes = 3, activation = None, in_channels = color_channels, encoder_weights=None)  
            torch.save(unet, modelsdir + 'Unet_' + encoder_name + '.pt')

    else:
        run_name = run_builder(aug_GaussianBlur, aug_ColorJitter, aug_HorizontalFlip, aug_Rotate, aug_RandomCrop, 
                encoder_name, optimizer, loss_function,
                True,False,False,False,False)
        unet_load,hist_load = torch.load("synthetic-moon-models/" + run_name + "/" + run_name + "_Unet.pt")
        unet = transfer_weights(unet_load, encoder)

    return unet

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------

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

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------

def plotinput(idxplot,df,IMAGE_PATH,MASK_PATH,figsdir,datasettype,test=True):

    if (datasettype == "synthetic-moon"):
        img = Image.open(IMAGE_PATH + 'render' + df['id'][idxplot] + '.png')
        mask = Image.open(MASK_PATH + 'ground' + df['id'][idxplot] + '.png')

    if (datasettype == "real-moon"):
        img = Image.open(IMAGE_PATH + df['id'][idxplot] + '.png')
        mask = Image.open(MASK_PATH + 'g_' + df['id'][idxplot] + '.png')

    if (datasettype == "ai4mars" or datasettype == "ai4mars-inference"):
        img = Image.open(IMAGE_PATH + df['id'][idxplot] + '.JPG')
        if (test): mask = Image.open(MASK_PATH + df['id'][idxplot] + '_merged.png')
        if (not test): mask = Image.open(MASK_PATH + df['id'][idxplot] + '.png')

    if (datasettype == "marsdataset3"):
        img = Image.open(IMAGE_PATH + df['id'][idxplot] + '.png')
        mask = Image.open(MASK_PATH + df['id'][idxplot] + '.png')

    print('Image Size', np.asarray(img).shape)
    print('Mask Size', np.asarray(mask).shape)

    #note the different colors
    plt.figure(num=1,clear=True)
    plt.imshow(img)
    plt.imshow(mask, alpha=0.3)
    plt.title('Picture with Mask Applied')
    plt.savefig(figsdir + "mask_%i.png" % idxplot)

    gc.collect()

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
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

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Mean and Standard Deviations

def getmeanstd(X_train, IMAGE_PATH, datasettype, comp = False):

    if (comp):

        n_train = len(X_train)

        #mean computation
        X_train_imgs_r = 0.
        X_train_imgs_g = 0.
        X_train_imgs_b = 0.

        #std dev computation
        X_train_imgs_r_sq = 0.
        X_train_imgs_g_sq = 0.
        X_train_imgs_b_sq = 0.

        n_tot = 0

        for i_train,x in enumerate(X_train):

            if (datasettype == "synthetic-moon"): img = cv2.imread(IMAGE_PATH + 'render' + x + '.png')
            if (datasettype == "real-moon"): img = cv2.imread(IMAGE_PATH + x + '.png')
            if (datasettype == "ai4mars" or datasettype == "ai4mars-inference"): img = cv2.imread(IMAGE_PATH + x + '.JPG')
            if (datasettype == "marsdataset3"): img = cv2.imread(IMAGE_PATH + 'train/' + x + '.png')

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #from BGR to RGB
            img = np.array(img, dtype = np.int64)

            a = np.shape(img) #not the same resolution for all
            
            X_train_imgs_r += np.sum(img[:,:,0])
            X_train_imgs_g += np.sum(img[:,:,1])
            X_train_imgs_b += np.sum(img[:,:,2])
            
            X_train_imgs_r_sq += np.sum(np.square(img[:,:,0]))
            X_train_imgs_g_sq += np.sum(np.square(img[:,:,1]))
            X_train_imgs_b_sq += np.sum(np.square(img[:,:,2]))

            n_tot += a[0]*a[1]
            
        mean = [X_train_imgs_r/(n_tot),X_train_imgs_g/(n_tot),X_train_imgs_b/(n_tot)]

        std_dev = [X_train_imgs_r_sq/(n_tot) - (mean[0]*mean[0]),
                X_train_imgs_g_sq/(n_tot) - (mean[1]*mean[1]),
                X_train_imgs_b_sq/(n_tot) - (mean[2]*mean[2])]

        std_dev = np.sqrt(std_dev)
        mean = np.divide(mean,255).tolist()
        std_dev = np.divide(std_dev,255).tolist()

        print(mean)
        print(std_dev)
    
    else:
        
        if (datasettype == "synthetic-moon"): 
            mean = [0.38694046508089297, 0.3868203630007706, 0.38693639003325914]
            std_dev = [0.2525432925470617, 0.25246713776664603, 0.2525414825486369]
        if (datasettype == "real-moon"): 
            mean = [0.3748141096978818, 0.35596510164602224, 0.2955574219124973]
            std_dev = [0.218745767389803, 0.2105994343046987, 0.18040353055725178]
        if (datasettype == "ai4mars" or datasettype == "ai4mars-inference"):
            mean = [0.2555624275034261, 0.2555624275034261, 0.2555624275034261]
            std_dev = [0.10463496527047811, 0.10463496527047811, 0.10463496527047811]
        if (datasettype == "marsdataset3"):
            mean = [0.6096051074840404, 0.5055833881828149, 0.35589452779742]
            std_dev = [0.14303784363117197, 0.12331537394970005, 0.0983118132001638]

    return mean,std_dev

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------


def getdistcts(train_set,val_set,test_set,datasettype,comp = False):

    if (comp): 

        # class frequency
        train_count = [0,0,0] #rocks, sky and terrain
        val_count = [0,0,0] #rocks, sky and terrain
        test_count = [0,0,0] #rocks, sky and terrain

        #all pixels
        pix_train = 0
        pix_val = 0
        pix_test = 0

        print("\n- TRAIN")
        for i_train,x in enumerate(train_set):
            
            if (i_train % (int(len(train_set)/10)) == 0): print("- %i / %i" % (i_train,len(train_set)))
            data = x[1]
            
            pix_train += data.shape[0]*data.shape[1]
            train_count[0] += np.count_nonzero(data == 0)
            train_count[1] += np.count_nonzero(data == 1)
            train_count[2] += np.count_nonzero(data == 2)
            
        train_count = np.divide(train_count,pix_train)

        print("\n- TEST")
        for i_test,x in enumerate(test_set):
            
            #if (i_test % (int(len(test_set)/10)) == 0): print("- %i / %i" % (i_test,len(test_set)))
            data = x[1]
            
            pix_test += data.shape[0]*data.shape[1]
            test_count[0] += np.count_nonzero(data == 0)
            test_count[1] += np.count_nonzero(data == 1)
            test_count[2] += np.count_nonzero(data == 2)

        test_count = np.divide(test_count,pix_test)

        print("\n- VALIDATION")
        for i_val,x in enumerate(val_set):
            
            #if (i_val % (int(len(val_set)/10)) == 0): print("- %i / %i" % (i_val,len(val_set)))
            data = x[1]
            
            pix_val += data.shape[0]*data.shape[1]
            val_count[0] += np.count_nonzero(data == 0)
            val_count[1] += np.count_nonzero(data == 1)
            val_count[2] += np.count_nonzero(data == 2)
            
        val_count = np.divide(val_count,pix_val)

        print(train_count,val_count,test_count)

    else:

        if (datasettype == "synthetic-moon"): 
            train_count = [0.06825831, 0.19470134, 0.73704035]
            val_count = [0.0683611,  0.18756708, 0.74407182]
            test_count = [0.06939824, 0.19352989, 0.73707187]

        if (datasettype == "real-moon"): 
            train_count = [0.06575352,0.25109662,0.68314986]
            val_count = [0.02721946,0.23215317,0.74062737]
            test_count = [0.03303425,0.20899641,0.75796934]

        if (datasettype == "ai4mars" or datasettype == "ai4mars-inference"):
            train_count = [0.00940106,0.14229407,0.84830487]
            val_count = [0.01033491,0.17835729,0.81130779]
            test_count = [0.0009083,0.16207275,0.83701894]

        if (datasettype == "marsdataset3"):
            train_count = [2.58378074e-01, 6.28637889e-04,7.40993288e-01] 
            val_count = [2.19322000e-01, 7.33012245e-04, 7.79944988e-01]
            test_count = [0.22812271, 0.00079636, 0.77108093]

    return train_count, val_count, test_count


def plot_freq(train_count, val_count, test_count, figsdir):

    rocks_count = [train_count[0],val_count[0],test_count[0]]
    sky_count = [train_count[1],val_count[1],test_count[1]]
    terrain_count = [train_count[2],val_count[2],test_count[2]]

    index = ['Train','Validation','Test']
    df_frequency = pd.DataFrame({'Rocks': rocks_count, 'Sky': sky_count, 'Terrain': terrain_count},index)
    df_frequency.plot.bar(stacked = True)
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.savefig(figsdir + "frequency_plot.png")

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

    # Make normalizer and formatter
    norm = matplotlib.colors.BoundaryNorm(norm_bins, len_lab, clip=True)
    fmt = matplotlib.ticker.FuncFormatter(lambda x, pos: labels[norm(x)])

    # Plot our figure
    fig, ax = plt.subplots()
    ax.imshow(image.permute(1, 2, 0),cmap="gray")
    im = ax.imshow(prediction.cpu().view(npixx,npixy), alpha = 0.6, cmap=cm, norm=norm)
    diff = norm_bins[1:] - norm_bins[:-1]
    tickz = norm_bins[:-1] + diff / 2
    cb = fig.colorbar(im, format=fmt, ticks=tickz, orientation='horizontal')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(figsdir + "pred_%i.pdf" % idata)
    plt.close('all')

    fig, ax = plt.subplots()
    ax.imshow(image.permute(1, 2, 0),cmap="gray")
    im = ax.imshow(mask.cpu().view(npixx,npixy), alpha = 0.6, cmap=cm, norm=norm)
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


def pixel_metrics(output, mask, metric = "acc"):
    
    with torch.no_grad():
            
        #applying softmax and choosing best class with argmax
        output = F.softmax(output, dim=1) #on all colors
        output = torch.argmax(output, dim=1) #on all colors

        #need 1d vectors for sklearn.metrics -> need cpu to convert to numpy 
        output = output.cpu().contiguous().view(-1) #1d
        mask = mask.cpu().contiguous().view(-1) #1d
        
    if (metric == "Accuracy"): return metrics.accuracy_score(mask,output)
    #if (metric == "Balanced Accuracy"): return metrics.balanced_accuracy_score(mask,output)
    if (metric == "Balanced Accuracy"): return balanced_accuracy_score(mask,output)
    if (metric == "Confusion Matrix"): return metrics.confusion_matrix(mask,output,labels = [0,1,2])
    if (metric == "Jaccard Score"): return metrics.jaccard_score(mask,output,average = "macro",zero_division = 1)
    else: 
        print("Metric not found...")
        return None

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Training Function

def trainbnorm(layer):
    layer.train() #set batchnorm to training mode
    for p in layer.parameters(): p.requires_grad_(True) #compute gradients

    # layer.track_running_stats = True
    # layer.weight.requires_grad_(True)
    # layer.bias.requires_grad_(True)

def trainconv(layer):
    for p in layer.parameters(): p.requires_grad_(True) #compute gradients
    # layer.weight.requires_grad_(True)
    # if bias: layer.bias.requires_grad_(True)

def model_train_18(model, ftuneenc, ftunedec, ftunebnorm, ftuneadapt):

    if (ftuneenc):

        #first block
        trainconv(model.encoder_1_1_conv)
        trainbnorm(model.encoder_1_2_batchnorm)
        
        #second block
        trainconv(model.encoder_2_1_conv)
        trainbnorm(model.encoder_2_2_batchnorm)
        trainconv(model.encoder_2_4_conv)
        trainbnorm(model.encoder_2_5_batchnorm)
        trainconv(model.encoder_2_7_conv)
        trainbnorm(model.encoder_2_8_batchnorm)
        trainconv(model.encoder_2_10_conv)
        trainbnorm(model.encoder_2_11_batchnorm)
        
        #third block
        trainconv(model.encoder_3_1_conv)
        trainbnorm(model.encoder_3_2_batchnorm)
        trainconv(model.encoder_3_4_conv)
        trainbnorm(model.encoder_3_5_batchnorm)
        trainconv(model.encoder_3_6_downsample)
        trainbnorm(model.encoder_3_7_batchnorm)
        trainconv(model.encoder_3_9_conv)
        trainbnorm(model.encoder_3_10_batchnorm)
        trainconv(model.encoder_3_12_conv)
        trainbnorm(model.encoder_3_13_batchnorm)

        #fourth block
        trainconv(model.encoder_4_1_conv)
        trainbnorm(model.encoder_4_2_batchnorm)
        trainconv(model.encoder_4_4_conv)
        trainbnorm(model.encoder_4_5_batchnorm)
        trainconv(model.encoder_4_6_downsample)
        trainbnorm(model.encoder_4_7_batchnorm)
        trainconv(model.encoder_4_9_conv)
        trainbnorm(model.encoder_4_10_batchnorm)
        trainconv(model.encoder_4_12_conv)
        trainbnorm(model.encoder_4_13_batchnorm)

        #fifth block
        trainconv(model.encoder_5_1_conv)
        trainbnorm(model.encoder_5_2_batchnorm)
        trainconv(model.encoder_5_4_conv)
        trainbnorm(model.encoder_5_5_batchnorm)
        trainconv(model.encoder_5_6_downsample)
        trainbnorm(model.encoder_5_7_batchnorm)
        trainconv(model.encoder_5_9_conv)
        trainbnorm(model.encoder_5_10_batchnorm)
        trainconv(model.encoder_5_12_conv)
        trainbnorm(model.encoder_5_13_batchnorm)

    if (ftunedec):

        #first block
        trainconv(model.decoder_1_1_conv)
        trainbnorm(model.decoder_1_2_batchnorm)
        trainconv(model.decoder_1_4_conv)
        trainbnorm(model.decoder_1_5_batchnorm)

        #second block
        trainconv(model.decoder_2_1_conv)
        trainbnorm(model.decoder_2_2_batchnorm)
        trainconv(model.decoder_2_4_conv)
        trainbnorm(model.decoder_2_5_batchnorm)

        #third block
        trainconv(model.decoder_3_1_conv)
        trainbnorm(model.decoder_3_2_batchnorm)
        trainconv(model.decoder_3_4_conv)
        trainbnorm(model.decoder_3_5_batchnorm)

        #fourth block
        trainconv(model.decoder_4_1_conv)
        trainbnorm(model.decoder_4_2_batchnorm)
        trainconv(model.decoder_4_4_conv)
        trainbnorm(model.decoder_4_5_batchnorm)

        #fifth block
        trainconv(model.decoder_5_1_conv)
        trainbnorm(model.decoder_5_2_batchnorm)
        trainconv(model.decoder_5_4_conv)
        trainbnorm(model.decoder_5_5_batchnorm)

        #seg head
        trainconv(model.decoder_6_1_conv)

    if (ftunebnorm):

        #first block
        trainbnorm(model.encoder_1_2_batchnorm)
        
        #second block
        trainbnorm(model.encoder_2_2_batchnorm)
        trainbnorm(model.encoder_2_5_batchnorm)
        trainbnorm(model.encoder_2_8_batchnorm)
        trainbnorm(model.encoder_2_11_batchnorm)
        
        #third block
        trainbnorm(model.encoder_3_2_batchnorm)
        trainbnorm(model.encoder_3_5_batchnorm)
        trainbnorm(model.encoder_3_7_batchnorm)
        trainbnorm(model.encoder_3_10_batchnorm)
        trainbnorm(model.encoder_3_13_batchnorm)

        #fourth block
        trainbnorm(model.encoder_4_2_batchnorm)
        trainbnorm(model.encoder_4_5_batchnorm)
        trainbnorm(model.encoder_4_7_batchnorm)
        trainbnorm(model.encoder_4_10_batchnorm)
        trainbnorm(model.encoder_4_13_batchnorm)

        #fifth block
        trainbnorm(model.encoder_5_2_batchnorm)
        trainbnorm(model.encoder_5_5_batchnorm)
        trainbnorm(model.encoder_5_7_batchnorm)
        trainbnorm(model.encoder_5_10_batchnorm)
        trainbnorm(model.encoder_5_13_batchnorm)

        #first block
        trainbnorm(model.decoder_1_2_batchnorm)
        trainbnorm(model.decoder_1_5_batchnorm)

        #second block
        trainbnorm(model.decoder_2_2_batchnorm)
        trainbnorm(model.decoder_2_5_batchnorm)

        #third block
        trainbnorm(model.decoder_3_2_batchnorm)
        trainbnorm(model.decoder_3_5_batchnorm)

        #fourth block
        trainbnorm(model.decoder_4_2_batchnorm)
        trainbnorm(model.decoder_4_5_batchnorm)

        #fifth block
        trainbnorm(model.decoder_5_2_batchnorm)
        trainbnorm(model.decoder_5_5_batchnorm)

    if (ftuneadapt):

        trainbnorm(model.adapter_1_1_batchnorm)
        trainconv(model.adapter_1_2_conv)
        trainbnorm(model.adapter_2_1_batchnorm)
        trainconv(model.adapter_2_2_conv)
        trainbnorm(model.adapter_3_1_batchnorm)
        trainconv(model.adapter_3_2_conv)
        trainbnorm(model.adapter_4_1_batchnorm)
        trainconv(model.adapter_4_2_conv)
        trainbnorm(model.adapter_5_1_batchnorm)
        trainconv(model.adapter_5_2_conv)
        trainbnorm(model.adapter_6_1_batchnorm)
        trainconv(model.adapter_6_2_conv)
        trainbnorm(model.adapter_7_1_batchnorm)
        trainconv(model.adapter_7_2_conv)
        trainbnorm(model.adapter_8_1_batchnorm)
        trainconv(model.adapter_8_2_conv)
        trainbnorm(model.adapter_9_1_batchnorm)
        trainconv(model.adapter_9_2_conv)
        trainbnorm(model.adapter_10_1_batchnorm)
        trainconv(model.adapter_10_2_conv)
        trainbnorm(model.adapter_11_1_batchnorm)
        trainconv(model.adapter_11_2_conv)
        trainbnorm(model.adapter_12_1_batchnorm)
        trainconv(model.adapter_12_2_conv)
        trainbnorm(model.adapter_13_1_batchnorm)
        trainconv(model.adapter_13_2_conv)
        trainbnorm(model.adapter_14_1_batchnorm)
        trainconv(model.adapter_14_2_conv)
        trainbnorm(model.adapter_15_1_batchnorm)
        trainconv(model.adapter_15_2_conv)
        trainbnorm(model.adapter_16_1_batchnorm)
        trainconv(model.adapter_16_2_conv)
        trainbnorm(model.adapter_17_1_batchnorm)
        trainconv(model.adapter_17_2_conv)
        trainbnorm(model.adapter_18_1_batchnorm)
        trainconv(model.adapter_18_2_conv)
        trainbnorm(model.adapter_19_1_batchnorm)
        trainconv(model.adapter_19_2_conv)
        trainbnorm(model.adapter_20_1_batchnorm)
        trainconv(model.adapter_20_2_conv)
        trainbnorm(model.adapter_21_1_batchnorm)
        trainconv(model.adapter_21_2_conv)
        trainbnorm(model.adapter_22_1_batchnorm)
        trainconv(model.adapter_22_2_conv)
        trainbnorm(model.adapter_23_1_batchnorm)
        trainconv(model.adapter_23_2_conv)
        trainbnorm(model.adapter_24_1_batchnorm)
        trainconv(model.adapter_24_2_conv)
        trainbnorm(model.adapter_25_1_batchnorm)
        trainconv(model.adapter_25_2_conv)
        trainbnorm(model.adapter_26_1_batchnorm)
        trainconv(model.adapter_26_2_conv)
        trainbnorm(model.adapter_27_1_batchnorm)
        trainconv(model.adapter_27_2_conv)

def model_train_vgg19bn(model, ftuneenc, ftunedec, ftunebnorm, ftuneadapt):

    if (ftuneenc):

        #first block
        trainconv(model.encoder_1_1_conv)
        trainbnorm(model.encoder_1_2_batchnorm)
        trainconv(model.encoder_1_4_conv)
        trainbnorm(model.encoder_1_5_batchnorm)
        
        #second block
        trainconv(model.encoder_2_1_conv)
        trainbnorm(model.encoder_2_2_batchnorm)
        trainconv(model.encoder_2_4_conv)
        trainbnorm(model.encoder_2_5_batchnorm)
        
        #third block
        trainconv(model.encoder_3_1_conv)
        trainbnorm(model.encoder_3_2_batchnorm)
        trainconv(model.encoder_3_4_conv)
        trainbnorm(model.encoder_3_5_batchnorm)
        trainconv(model.encoder_3_7_conv)
        trainbnorm(model.encoder_3_8_batchnorm)
        trainconv(model.encoder_3_10_conv)
        trainbnorm(model.encoder_3_11_batchnorm)

        #fourth block
        trainconv(model.encoder_4_1_conv)
        trainbnorm(model.encoder_4_2_batchnorm)
        trainconv(model.encoder_4_4_conv)
        trainbnorm(model.encoder_4_5_batchnorm)
        trainconv(model.encoder_4_7_conv)
        trainbnorm(model.encoder_4_8_batchnorm)
        trainconv(model.encoder_4_10_conv)
        trainbnorm(model.encoder_4_11_batchnorm)

        #fifth block
        trainconv(model.encoder_5_1_conv)
        trainbnorm(model.encoder_5_2_batchnorm)
        trainconv(model.encoder_5_4_conv)
        trainbnorm(model.encoder_5_5_batchnorm)
        trainconv(model.encoder_5_7_conv)
        trainbnorm(model.encoder_5_8_batchnorm)
        trainconv(model.encoder_5_10_conv)
        trainbnorm(model.encoder_5_11_batchnorm)

    if (ftunedec):

        #first block
        trainconv(model.decoder_1_1_conv)
        trainbnorm(model.decoder_1_2_batchnorm)
        trainconv(model.decoder_1_4_conv)
        trainbnorm(model.decoder_1_5_batchnorm)

        #second block
        trainconv(model.decoder_2_1_conv)
        trainbnorm(model.decoder_2_2_batchnorm)
        trainconv(model.decoder_2_4_conv)
        trainbnorm(model.decoder_2_5_batchnorm)

        #third block
        trainconv(model.decoder_3_1_conv)
        trainbnorm(model.decoder_3_2_batchnorm)
        trainconv(model.decoder_3_4_conv)
        trainbnorm(model.decoder_3_5_batchnorm)

        #fourth block
        trainconv(model.decoder_4_1_conv)
        trainbnorm(model.decoder_4_2_batchnorm)
        trainconv(model.decoder_4_4_conv)
        trainbnorm(model.decoder_4_5_batchnorm)

        #fifth block
        trainconv(model.decoder_5_1_conv)
        trainbnorm(model.decoder_5_2_batchnorm)
        trainconv(model.decoder_5_4_conv)
        trainbnorm(model.decoder_5_5_batchnorm)

        #sixth block
        trainconv(model.decoder_6_1_conv)
        trainbnorm(model.decoder_6_2_batchnorm)
        trainconv(model.decoder_6_4_conv)
        trainbnorm(model.decoder_6_5_batchnorm)

        #segmentation head
        trainconv(model.decoder_7_1_conv)

    if (ftunebnorm):

        #first block
        trainbnorm(model.encoder_1_2_batchnorm)
        trainbnorm(model.encoder_1_5_batchnorm)
        
        #second block
        trainbnorm(model.encoder_2_2_batchnorm)
        trainbnorm(model.encoder_2_5_batchnorm)
        
        #third block
        trainbnorm(model.encoder_3_2_batchnorm)
        trainbnorm(model.encoder_3_5_batchnorm)
        trainbnorm(model.encoder_3_8_batchnorm)
        trainbnorm(model.encoder_3_11_batchnorm)

        #fourth block
        trainbnorm(model.encoder_4_2_batchnorm)
        trainbnorm(model.encoder_4_5_batchnorm)
        trainbnorm(model.encoder_4_8_batchnorm)
        trainbnorm(model.encoder_4_11_batchnorm)

        #fifth block
        trainbnorm(model.encoder_5_2_batchnorm)
        trainbnorm(model.encoder_5_5_batchnorm)
        trainbnorm(model.encoder_5_8_batchnorm)
        trainbnorm(model.encoder_5_11_batchnorm)

        #first block
        trainbnorm(model.decoder_1_2_batchnorm)
        trainbnorm(model.decoder_1_5_batchnorm)

        #second block
        trainbnorm(model.decoder_2_2_batchnorm)
        trainbnorm(model.decoder_2_5_batchnorm)

        #third block
        trainbnorm(model.decoder_3_2_batchnorm)
        trainbnorm(model.decoder_3_5_batchnorm)

        #fourth block
        trainbnorm(model.decoder_4_2_batchnorm)
        trainbnorm(model.decoder_4_5_batchnorm)

        #fifth block
        trainbnorm(model.decoder_5_2_batchnorm)
        trainbnorm(model.decoder_5_5_batchnorm)

        #sixth block
        trainbnorm(model.decoder_6_2_batchnorm)
        trainbnorm(model.decoder_6_5_batchnorm)

    if (ftuneadapt):

        trainbnorm(model.adapter_1_1_batchnorm)
        trainconv(model.adapter_1_2_conv)
        trainbnorm(model.adapter_2_1_batchnorm)
        trainconv(model.adapter_2_2_conv)
        trainbnorm(model.adapter_3_1_batchnorm)
        trainconv(model.adapter_3_2_conv)
        trainbnorm(model.adapter_4_1_batchnorm)
        trainconv(model.adapter_4_2_conv)
        trainbnorm(model.adapter_5_1_batchnorm)
        trainconv(model.adapter_5_2_conv)
        trainbnorm(model.adapter_6_1_batchnorm)
        trainconv(model.adapter_6_2_conv)
        trainbnorm(model.adapter_7_1_batchnorm)
        trainconv(model.adapter_7_2_conv)
        trainbnorm(model.adapter_8_1_batchnorm)
        trainconv(model.adapter_8_2_conv)
        trainbnorm(model.adapter_9_1_batchnorm)
        trainconv(model.adapter_9_2_conv)
        trainbnorm(model.adapter_10_1_batchnorm)
        trainconv(model.adapter_10_2_conv)
        trainbnorm(model.adapter_11_1_batchnorm)
        trainconv(model.adapter_11_2_conv)
        trainbnorm(model.adapter_12_1_batchnorm)
        trainconv(model.adapter_12_2_conv)
        trainbnorm(model.adapter_13_1_batchnorm)
        trainconv(model.adapter_13_2_conv)
        trainbnorm(model.adapter_14_1_batchnorm)
        trainconv(model.adapter_14_2_conv)
        trainbnorm(model.adapter_15_1_batchnorm)
        trainconv(model.adapter_15_2_conv)
        trainbnorm(model.adapter_16_1_batchnorm)
        trainconv(model.adapter_16_2_conv)
        trainbnorm(model.adapter_17_1_batchnorm)
        trainconv(model.adapter_17_2_conv)
        trainbnorm(model.adapter_18_1_batchnorm)
        trainconv(model.adapter_18_2_conv)
        trainbnorm(model.adapter_19_1_batchnorm)
        trainconv(model.adapter_19_2_conv)
        trainbnorm(model.adapter_20_1_batchnorm)
        trainconv(model.adapter_20_2_conv)
        trainbnorm(model.adapter_21_1_batchnorm)
        trainconv(model.adapter_21_2_conv)
        trainbnorm(model.adapter_22_1_batchnorm)
        trainconv(model.adapter_22_2_conv)
        trainbnorm(model.adapter_23_1_batchnorm)
        trainconv(model.adapter_23_2_conv)
        trainbnorm(model.adapter_24_1_batchnorm)
        trainconv(model.adapter_24_2_conv)
        trainbnorm(model.adapter_25_1_batchnorm)
        trainconv(model.adapter_25_2_conv)
        trainbnorm(model.adapter_26_1_batchnorm)
        trainconv(model.adapter_26_2_conv)
        trainbnorm(model.adapter_27_1_batchnorm)
        trainconv(model.adapter_27_2_conv)
        trainbnorm(model.adapter_28_1_batchnorm)
        trainconv(model.adapter_28_2_conv)

def model_train_18_rep2(model, ftuneenc, ftunedec, ftunebnorm, ftuneadapt):

    if (ftuneenc):

        #first block
        trainconv(model.encoder_1_1_conv)
        trainbnorm(model.encoder_1_2_batchnorm)
        
        #second block
        trainconv(model.encoder_2_1_conv)
        trainbnorm(model.encoder_2_2_batchnorm)
        trainconv(model.encoder_2_4_conv)
        trainbnorm(model.encoder_2_5_batchnorm)
        trainconv(model.encoder_2_7_conv)
        trainbnorm(model.encoder_2_8_batchnorm)
        trainconv(model.encoder_2_10_conv)
        trainbnorm(model.encoder_2_11_batchnorm)
        
        #third block
        trainconv(model.encoder_3_1_conv)
        trainbnorm(model.encoder_3_2_batchnorm)
        trainconv(model.encoder_3_4_conv)
        trainbnorm(model.encoder_3_5_batchnorm)
        trainconv(model.encoder_3_6_downsample)
        trainbnorm(model.encoder_3_7_batchnorm)
        trainconv(model.encoder_3_9_conv)
        trainbnorm(model.encoder_3_10_batchnorm)
        trainconv(model.encoder_3_12_conv)
        trainbnorm(model.encoder_3_13_batchnorm)

        #fourth block
        trainconv(model.encoder_4_1_conv)
        trainbnorm(model.encoder_4_2_batchnorm)
        trainconv(model.encoder_4_4_conv)
        trainbnorm(model.encoder_4_5_batchnorm)
        trainconv(model.encoder_4_6_downsample)
        trainbnorm(model.encoder_4_7_batchnorm)
        trainconv(model.encoder_4_9_conv)
        trainbnorm(model.encoder_4_10_batchnorm)
        trainconv(model.encoder_4_12_conv)
        trainbnorm(model.encoder_4_13_batchnorm)

        #fifth block
        trainconv(model.encoder_5_1_conv)
        trainbnorm(model.encoder_5_2_batchnorm)
        trainconv(model.encoder_5_4_conv)
        trainbnorm(model.encoder_5_5_batchnorm)
        trainconv(model.encoder_5_6_downsample)
        trainbnorm(model.encoder_5_7_batchnorm)
        trainconv(model.encoder_5_9_conv)
        trainbnorm(model.encoder_5_10_batchnorm)
        trainconv(model.encoder_5_12_conv)
        trainbnorm(model.encoder_5_13_batchnorm)

    if (ftunedec):

        #first block
        trainconv(model.decoder_1_1_conv)
        trainbnorm(model.decoder_1_2_batchnorm)
        trainconv(model.decoder_1_4_conv)
        trainbnorm(model.decoder_1_5_batchnorm)

        #second block
        trainconv(model.decoder_2_1_conv)
        trainbnorm(model.decoder_2_2_batchnorm)
        trainconv(model.decoder_2_4_conv)
        trainbnorm(model.decoder_2_5_batchnorm)

        #third block
        trainconv(model.decoder_3_1_conv)
        trainbnorm(model.decoder_3_2_batchnorm)
        trainconv(model.decoder_3_4_conv)
        trainbnorm(model.decoder_3_5_batchnorm)

        #fourth block
        trainconv(model.decoder_4_1_conv)
        trainbnorm(model.decoder_4_2_batchnorm)
        trainconv(model.decoder_4_4_conv)
        trainbnorm(model.decoder_4_5_batchnorm)

        #fifth block
        trainconv(model.decoder_5_1_conv)
        trainbnorm(model.decoder_5_2_batchnorm)
        trainconv(model.decoder_5_4_conv)
        trainbnorm(model.decoder_5_5_batchnorm)

        #seg head
        trainconv(model.decoder_6_1_conv)

    if (ftunebnorm):

        #first block
        trainbnorm(model.encoder_1_2_batchnorm)
        
        #second block
        trainbnorm(model.encoder_2_2_batchnorm)
        trainbnorm(model.encoder_2_5_batchnorm)
        trainbnorm(model.encoder_2_8_batchnorm)
        trainbnorm(model.encoder_2_11_batchnorm)
        
        #third block
        trainbnorm(model.encoder_3_2_batchnorm)
        trainbnorm(model.encoder_3_5_batchnorm)
        trainbnorm(model.encoder_3_7_batchnorm)
        trainbnorm(model.encoder_3_10_batchnorm)
        trainbnorm(model.encoder_3_13_batchnorm)

        #fourth block
        trainbnorm(model.encoder_4_2_batchnorm)
        trainbnorm(model.encoder_4_5_batchnorm)
        trainbnorm(model.encoder_4_7_batchnorm)
        trainbnorm(model.encoder_4_10_batchnorm)
        trainbnorm(model.encoder_4_13_batchnorm)

        #fifth block
        trainbnorm(model.encoder_5_2_batchnorm)
        trainbnorm(model.encoder_5_5_batchnorm)
        trainbnorm(model.encoder_5_7_batchnorm)
        trainbnorm(model.encoder_5_10_batchnorm)
        trainbnorm(model.encoder_5_13_batchnorm)

        #first block
        trainbnorm(model.decoder_1_2_batchnorm)
        trainbnorm(model.decoder_1_5_batchnorm)

        #second block
        trainbnorm(model.decoder_2_2_batchnorm)
        trainbnorm(model.decoder_2_5_batchnorm)

        #third block
        trainbnorm(model.decoder_3_2_batchnorm)
        trainbnorm(model.decoder_3_5_batchnorm)

        #fourth block
        trainbnorm(model.decoder_4_2_batchnorm)
        trainbnorm(model.decoder_4_5_batchnorm)

        #fifth block
        trainbnorm(model.decoder_5_2_batchnorm)
        trainbnorm(model.decoder_5_5_batchnorm)

    if (ftuneadapt):

        trainbnorm(model.adapter_1_1_batchnorm)
        trainconv(model.adapter_1_2_conv)
        trainbnorm(model.adapter_2_1_batchnorm)
        trainconv(model.adapter_2_2_conv)
        trainbnorm(model.adapter_3_1_batchnorm)
        trainconv(model.adapter_3_2_conv)
        trainbnorm(model.adapter_4_1_batchnorm)
        trainconv(model.adapter_4_2_conv)
        trainbnorm(model.adapter_5_1_batchnorm)
        trainconv(model.adapter_5_2_conv)
        trainbnorm(model.adapter_6_1_batchnorm)
        trainconv(model.adapter_6_2_conv)
        trainbnorm(model.adapter_7_1_batchnorm)
        trainconv(model.adapter_7_2_conv)
        trainbnorm(model.adapter_8_1_batchnorm)
        trainconv(model.adapter_8_2_conv)
        trainbnorm(model.adapter_9_1_batchnorm)
        trainconv(model.adapter_9_2_conv)
        trainbnorm(model.adapter_10_1_batchnorm)
        trainconv(model.adapter_10_2_conv)
        trainbnorm(model.adapter_11_1_batchnorm)
        trainconv(model.adapter_11_2_conv)
        trainbnorm(model.adapter_12_1_batchnorm)
        trainconv(model.adapter_12_2_conv)
        trainbnorm(model.adapter_13_1_batchnorm)
        trainconv(model.adapter_13_2_conv)
        trainbnorm(model.adapter_14_1_batchnorm)
        trainconv(model.adapter_14_2_conv)
        trainbnorm(model.adapter_15_1_batchnorm)
        trainconv(model.adapter_15_2_conv)
        trainbnorm(model.adapter_16_1_batchnorm)
        trainconv(model.adapter_16_2_conv)
        trainbnorm(model.adapter_17_1_batchnorm)
        trainconv(model.adapter_17_2_conv)
        trainbnorm(model.adapter_18_1_batchnorm)
        trainconv(model.adapter_18_2_conv)
        trainbnorm(model.adapter_19_1_batchnorm)
        trainconv(model.adapter_19_2_conv)
        trainbnorm(model.adapter_20_1_batchnorm)
        trainconv(model.adapter_20_2_conv)
        trainbnorm(model.adapter_21_1_batchnorm)
        trainconv(model.adapter_21_2_conv)
        trainbnorm(model.adapter_22_1_batchnorm)
        trainconv(model.adapter_22_2_conv)
        trainbnorm(model.adapter_23_1_batchnorm)
        trainconv(model.adapter_23_2_conv)
        trainbnorm(model.adapter_24_1_batchnorm)
        trainconv(model.adapter_24_2_conv)
        trainbnorm(model.adapter_25_1_batchnorm)
        trainconv(model.adapter_25_2_conv)
        trainbnorm(model.adapter_26_1_batchnorm)
        trainconv(model.adapter_26_2_conv)
        trainbnorm(model.adapter_27_1_batchnorm)
        trainconv(model.adapter_27_2_conv)

        trainbnorm(model.adapter_1_3_batchnorm)
        trainconv(model.adapter_1_4_conv)
        trainbnorm(model.adapter_2_3_batchnorm)
        trainconv(model.adapter_2_4_conv)
        trainbnorm(model.adapter_3_3_batchnorm)
        trainconv(model.adapter_3_4_conv)
        trainbnorm(model.adapter_4_3_batchnorm)
        trainconv(model.adapter_4_4_conv)
        trainbnorm(model.adapter_5_3_batchnorm)
        trainconv(model.adapter_5_4_conv)
        trainbnorm(model.adapter_6_3_batchnorm)
        trainconv(model.adapter_6_4_conv)
        trainbnorm(model.adapter_7_3_batchnorm)
        trainconv(model.adapter_7_4_conv)
        trainbnorm(model.adapter_8_3_batchnorm)
        trainconv(model.adapter_8_4_conv)
        trainbnorm(model.adapter_9_3_batchnorm)
        trainconv(model.adapter_9_4_conv)
        trainbnorm(model.adapter_10_3_batchnorm)
        trainconv(model.adapter_10_4_conv)
        trainbnorm(model.adapter_11_3_batchnorm)
        trainconv(model.adapter_11_4_conv)
        trainbnorm(model.adapter_12_3_batchnorm)
        trainconv(model.adapter_12_4_conv)
        trainbnorm(model.adapter_13_3_batchnorm)
        trainconv(model.adapter_13_4_conv)
        trainbnorm(model.adapter_14_3_batchnorm)
        trainconv(model.adapter_14_4_conv)
        trainbnorm(model.adapter_15_3_batchnorm)
        trainconv(model.adapter_15_4_conv)
        trainbnorm(model.adapter_16_3_batchnorm)
        trainconv(model.adapter_16_4_conv)
        trainbnorm(model.adapter_17_3_batchnorm)
        trainconv(model.adapter_17_4_conv)
        trainbnorm(model.adapter_18_3_batchnorm)
        trainconv(model.adapter_18_4_conv)
        trainbnorm(model.adapter_19_3_batchnorm)
        trainconv(model.adapter_19_4_conv)
        trainbnorm(model.adapter_20_3_batchnorm)
        trainconv(model.adapter_20_4_conv)
        trainbnorm(model.adapter_21_3_batchnorm)
        trainconv(model.adapter_21_4_conv)
        trainbnorm(model.adapter_22_3_batchnorm)
        trainconv(model.adapter_22_4_conv)
        trainbnorm(model.adapter_23_3_batchnorm)
        trainconv(model.adapter_23_4_conv)
        trainbnorm(model.adapter_24_3_batchnorm)
        trainconv(model.adapter_24_4_conv)
        trainbnorm(model.adapter_25_3_batchnorm)
        trainconv(model.adapter_25_4_conv)
        trainbnorm(model.adapter_26_3_batchnorm)
        trainconv(model.adapter_26_4_conv)
        trainbnorm(model.adapter_27_3_batchnorm)
        trainconv(model.adapter_27_4_conv)

def model_train(model, encoder, baseline, ftuneenc, ftunedec, ftunebnorm, ftuneadapt):

    if (baseline): model.train()
    else:
        model.eval() #setting batchnorm to evaluation
        for p in model.parameters(): p.requires_grad_(False) #freezing all layers
        if (encoder == "resnet18"): model_train_18(model, ftuneenc, ftunedec, ftunebnorm, ftuneadapt)
        if (encoder == "resnet18_relu"): model_train_18(model, ftuneenc, ftunedec, ftunebnorm, ftuneadapt)
        if (encoder == "resnet18_conv"): model_train_18(model, ftuneenc, ftunedec, ftunebnorm, ftuneadapt)
        if (encoder == "resnet18_rep2"): model_train_18_rep2(model, ftuneenc, ftunedec, ftunebnorm, ftuneadapt)
        if (encoder == "vgg19_bn"): model_train_vgg19bn(model, ftuneenc, ftunedec, ftunebnorm, ftuneadapt)

def fit(model, train_loader, val_loader, criterion, optimizer, scheduler, metric_names, 
        baseline, ftuneenc, ftunedec, ftunebnorm, ftuneadapt, encoder, 
        wandbopt, run, device, stopat = None):
    
    torch.cuda.empty_cache()
    
    train_losses = []
    test_losses = []
    
    train_metrics = []
    test_metrics = []
    
    lrs = [] #list of all learning rates for each epoch e.g. [[0.001,0.005,0.01],[0.0009,0.004,0.009],...]
    
    min_loss = np.inf
    decrease = 1
    not_improve = 0

    model.to(device)
    fit_time = time.time()

    epoch=0
    initial_lr=get_lr(optimizer)
    
    # for e in range(epochs):
    while True:
        epoch += 1

        since = time.time()
        running_loss = 0
        
        train_metr_epoch = [0,0,0]
        test_metr_epoch = [0,0,0]
        lr_list = [] #learning rate for each epoch
        
        #training loop
        model_train(model, encoder, baseline, ftuneenc, ftunedec, ftunebnorm, ftuneadapt)
        
        for i, data in enumerate(tqdm(train_loader)):
            
            if (stopat is not None and i == stopat): break
            
            #training phase
            image, mask = data
            
            image = image.to(device)
            mask = mask.to(device)
            
            #forward
            output = model(image)
            
            torch.use_deterministic_algorithms(False)
            lossfn = criterion
            loss = lossfn(output, mask)
            torch.use_deterministic_algorithms(True)       
                       
            #evaluation metrics
            for imetr,metr in enumerate(metric_names):
                train_metr_epoch[imetr] += pixel_metrics(output,mask,metr)
            
            #backward
            loss.backward()

            optimizer.step() #update weight          
            optimizer.zero_grad() #reset gradient
            
            #step the learning rate
            if (i==0): lr_list.append(get_lr(optimizer)) #need it only for first element of dataloader
                    
            running_loss += loss.item()
           
        #end training 

        #start evaluation
        model.eval()
        test_loss = 0.
        
        #validation loop
        with torch.no_grad():
            
            for i, data in enumerate(tqdm(val_loader)):
                
                if (stopat is not None and i == stopat): break
                
                image, mask = data
                image = image.to(device)
                mask = mask.to(device)
                
                #forward
                output = model(image)
 
                torch.use_deterministic_algorithms(False)
                lossfn = criterion
                loss = lossfn(output, mask)
                torch.use_deterministic_algorithms(True)       
 
                
                #evaluation metrics
                for imetr,metr in enumerate(metric_names):
                    test_metr_epoch[imetr] += pixel_metrics(output,mask,metr)
                    
                    
                test_loss += loss.item()
                
        #calculatio mean for each batch
        train_losses.append(running_loss/len(train_loader))
        test_losses.append(test_loss/len(val_loader))

        train_metrics.append(np.divide(train_metr_epoch,len(train_loader)).tolist())
        test_metrics.append(np.divide(test_metr_epoch,len(val_loader)).tolist())

        if min_loss > (test_loss/len(val_loader)):
            
            print('Loss Decreasing for {:}-th Time {:.3f} >> {:.3f} '.format(decrease, min_loss, (test_loss/len(val_loader))))
            min_loss = (test_loss/len(val_loader))
            decrease += 1

            not_improve=0
        
            #saving every 5 decrease
            #if decrease % 5 == 0:
            #    print('Saving model...')
            #    torch.save(model, 'Unet-{decrease}.pt')

        if (test_loss/len(val_loader)) > min_loss:
            
            not_improve += 1
            min_loss = (test_loss/len(val_loader))
            print('Loss Not Decreasing for {}-th Time'.format(not_improve))
            
            # if not_improve == 10:
            #     print('Loss not decreasing for 10 times, Cutting learning rate by 10:')
            #     lrs = lrs/10



                
        print("Epoch:{}..".format(epoch),
              "Train Loss: {:.3f}..".format(running_loss/len(train_loader)),
              "Val Loss: {:.3f}..".format(test_loss/len(val_loader)))
        
        for imetr,metr in enumerate(metric_names):
            print("Train " + metr + ": {:.3f}".format(train_metrics[epoch-1][imetr]))
            print("Val " + metr + ": {:.3f}".format(test_metrics[epoch-1][imetr]))
            
            
        #step the learning rate
        lrs.append(lr_list)

        #scheduler step at each epoch
        scheduler.step(test_loss/len(val_loader))

        if get_lr(optimizer)<(initial_lr/1000):
            break   

        # wandb logging things   
        if (wandbopt):

            print("Logging variables...")
            
            run.log({'epoch':epoch, 'learning_rate': lrs[-1][0],
                    'train_loss' : train_losses[-1], 'val_loss': test_losses[-1],
                    "Accuracy_train":train_metrics[-1][0], "Accuracy_val":test_metrics[-1][0],
                    "Balanced_Accuracy_train":train_metrics[-1][1], "Balanced_Accuracy_val":test_metrics[-1][1],
                    "Jaccard_train":train_metrics[-1][2], "Jaccard_val":test_metrics[-1][2]
                    })       






    history = {'train_loss' : train_losses, 'val_loss': test_losses,
               'train_metric' : train_metrics, 'val_metric': test_metrics,
               'lrs': lrs}
   
    
    print('Total time: {:.2f} m' .format((time.time()- fit_time)/60))
    return history

#find optimal tradeoff
def adaptertradeoff(unet_baseline, dataloader, encoder, method, metric_names, varpercentage, device):

    #maximum number of adapters
    if (encoder == "resnet18"): nadaptersmax = 27
    if (encoder == "vgg19_bn"): nadaptersmax = 28

    #evaluate performance with all adapters
    scores, conf_matrix = evaluate(unet_baseline,dataloader,metric_names,device)
    bal_acc_past = np.mean(scores[1])
    ibest = 0

    #algorithm for best tradeoff
    for iadapt in range(1,nadaptersmax+1):

        #remove adapters
        unet_remove = remove_adapters(unet_baseline, iadapt, encoder, method)
        unet_remove[0].eval()
        scores, conf_matrix = evaluate(unet_remove[0],dataloader,metric_names,device)
        bal_acc = np.mean(scores[1])
        varperc = (bal_acc - bal_acc_past)/bal_acc_past
        print("Removing %i adapters: %.5f percentage, %.4f delta, %.4f past, %.4f new" % (iadapt,varperc,bal_acc-bal_acc_past,bal_acc_past,bal_acc))
        if varperc < -varpercentage: 
            ibest = iadapt - 1
            break
        bal_acc_past = bal_acc

    return ibest



# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------


def plot_loss(history,nameloss,dirplot,figsdir):
    
    plt.figure(num=1,clear=True)
    plt.plot(history['val_loss'], label='Validation Loss', marker='o', color = "blue", linestyle = "dashed")
    plt.plot(history['train_loss'], label='Training Loss', marker='o', color = "green",linestyle = "dashed")
    plt.title('Loss per epoch')
    plt.ylabel(nameloss)
    plt.xlabel('Epoch')
    plt.legend(loc = "best")
    plt.grid()
    plt.savefig(figsdir + dirplot + "loss.png")

    gc.collect()

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------


def plot_score(history,metric_names,dirplot,figsdir):
    
    for imetr,metr in enumerate(metric_names):
        
        train_score = [history['train_metric'][e][imetr] for e in range(len(history['train_metric']))] 
        val_score = [history['val_metric'][e][imetr] for e in range(len(history['val_metric']))] 
        
        plt.figure(num=1,clear=True)
        plt.plot(train_score, label = "Training " + metr, marker='o', color = "green", linestyle = "dashed")
        plt.plot(val_score, label = "Validation " + metr, marker='o', color = "blue",linestyle = "dashed")
        plt.title('Score per epoch')
        plt.ylabel("Mean " + metr)
        plt.xlabel('Epoch')
        plt.legend(loc = "best")
        plt.grid()
        plt.savefig(figsdir + dirplot + "metr_%i.png" % imetr)

        gc.collect()

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------


def plot_lrs(history,dirplot,figsdir):
    
    lr_epoch = [history['lrs'][e] for e in range(len(history['lrs']))]
        
    plt.figure(num=1,clear=True)
    plt.plot(lr_epoch, marker='o', color = "b", linestyle = "dashed")
    plt.title('Learning Rates per epoch')
    plt.ylabel("Learning Rate")
    plt.xlabel('Epoch')
    plt.grid()
    plt.savefig(figsdir + dirplot + "lrs.png")

    gc.collect()

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------


def evaluate(model,test_loader,metric_names,device_eval):

    model.eval()
    model.to(device_eval)
    confusion_matrix = np.zeros((3,3))
    test_metrics = [[],[],[]]

    with torch.no_grad():    
        
        for i, data in enumerate(tqdm(test_loader)):
            
            image, mask = data
            image = image.to(device_eval)
            mask = mask.to(device_eval)
            
            #image = image.unsqueeze(0)
            #mask = mask.unsqueeze(0)

            #forward
            output = model(image)  

            #evaluation metrics
            for imetr,metr in enumerate(metric_names):
                test_metrics[imetr].append(pixel_metrics(output,mask,metr))

            if (np.array(pixel_metrics(output,mask,"Confusion Matrix").shape != (3,3))): 
                print(np.array(pixel_metrics(output,mask,"Confusion Matrix")))
            
            
            confusion_matrix += np.array(pixel_metrics(output,mask,"Confusion Matrix"))
    
    #confusion matrix normalization
    sum_of_rows = np.sum(confusion_matrix,axis = 1)
    confusion_matrix[:,:] = confusion_matrix[:,:] / sum_of_rows[:, np.newaxis]
    
    return test_metrics, confusion_matrix

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------


#functions to compare results
def plot_loss_compare(histories,labels,nameloss,dirplot,lasts = None):
    
    colors = ["r","g","b","c","m","y"]
    
    plt.figure(num=1,clear=True)
    for ih,history in enumerate(histories):
        
        n_epochs_plot = len(history['val_loss'])
        start_epoch = 0
        if (lasts): start_epoch = n_epochs_plot - lasts
        plt.plot(history['val_loss'][start_epoch:n_epochs_plot], 
                 label = labels[ih], 
                 marker='o', 
                 color = colors[ih], 
                 linestyle = "dashed")
    
    plt.title('Validation Loss per epoch')
    plt.ylabel(nameloss)
    plt.xlabel('Epoch')
    plt.legend(loc = "best")
    plt.grid()
    plt.savefig("figs/" + dirplot + "loss_compare.png")

    gc.collect()

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------


def plot_score_compare(histories,labels,metric_names,dirplot,lasts = None):
    
    colors = ["r","g","b","c","m","y"]
    
    for imetr,metr in enumerate(metric_names):
        
        plt.figure(num=1,clear=True)
        for ih,history in enumerate(histories): 
            
            n_epochs_plot = len(history['val_metric'])
            start_epoch = 0
            if (lasts): start_epoch = n_epochs_plot - lasts
            val_score = [history['val_metric'][e][imetr] for e in range(start_epoch,n_epochs_plot)] 
            plt.plot(val_score, label = labels[ih], marker='o', color = colors[ih],linestyle = "dashed")
        
        plt.title('Validation ' + metr + ' per epoch')
        plt.ylabel("Mean " + metr)
        plt.xlabel('Epoch')
        plt.legend(loc = "best")
        plt.grid()
        plt.savefig("figs/" + dirplot + "metr_%i_compare.png" % imetr)

        gc.collect()

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------

def predicted_mask(model,image,device_eval):
    
    image = image.to(device_eval)
    output = model(image)
    
    #applying softmax and choosing best class with argmax
    output = F.softmax(output, dim=1) #on all colors
    output = torch.argmax(output, dim=1) #on all colors
    
    return output

def plot_image_mask(dataset,idxplot,figsdir):

    data=dataset[idxplot]
    image = data[0]
    mask = data[1]    

    plt.figure(num=1,clear=True)
    plt.title("Image with Merged Mask")
    #plt.imshow(image.permute(1, 2, 0), cmap = "gray", vmin = 0, vmax = 255)
    plt.imshow(image.permute(1, 2, 0), cmap='gray')
    plt.imshow(mask, alpha = 0.1, cmap = "brg", vmin = 0, vmax = 2)
    plt.savefig(figsdir + "mask_merged_%i.png" % idxplot)

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Dataset Class

class FakeMoonDataset(Dataset):
    
    def __init__(self, img_path, mask_path, X, mean, std, transform = None, black_white = True):
        self.img_path = img_path
        self.mask_path = mask_path
        self.X = X
        self.transform = transform
        self.mean = mean
        self.std = std
        self.black_white = black_white
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        
        #lettura immagini
        img = cv2.imread(self.img_path + 'render' + self.X[idx] + '.png')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #from BGR to RGB
        mask = cv2.imread(self.mask_path + 'ground' + self.X[idx] + '.png')
        
        #merging mask
        mask = self.merge_moon_synth(mask)
        
        #converting to b/w labels
        mask = self.mask_to_label(mask)
        
        #transformations
        if self.transform is not None:
            aug = self.transform(image=img, mask=mask)
            img = Image.fromarray(aug['image'])
            mask = aug['mask']
        
        if self.transform is None: img = Image.fromarray(img)
        
        #normalization
        if (self.black_white): t = T.Compose([T.ToTensor(), T.Normalize(self.mean, self.std), T.Grayscale()])
        else: t = T.Compose([T.ToTensor(), T.Normalize(self.mean, self.std)])
        
        img = t(img)
        mask = torch.from_numpy(mask).long()
            
        return img, mask
    
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
    
    def __init__(self, img_path, mask_path, X, mean, std, transform = None, black_white = True):
        self.img_path = img_path
        self.mask_path = mask_path
        self.X = X
        self.transform = transform
        self.mean = mean
        self.std = std
        self.black_white = black_white
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        
        #lettura immagini
        img = cv2.imread(self.img_path + self.X[idx] + '.png')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #from BGR to RGB
        mask = cv2.imread(self.mask_path + 'g_' + self.X[idx] + '.png')

        # if (img.shape != mask.shape):
        #     print('Warning! -> Mask and Image are not the same')
        #     print(self.X[idx])
        #     print(img.shape)
        #     print(mask.shape)
        #     resizeimg = A.Resize(mask.shape[0],mask.shape[1],always_apply=True,p=1)
        #     img = resizeimg(image = img)
        #     img = img['image']

        #merging mask
        mask = self.merge_moon_synth(mask)
        
        #converting to b/w labels
        mask = self.mask_to_label(mask)
        
        #transformations
        if self.transform is not None:
            aug = self.transform(image=img, mask=mask)
            img = Image.fromarray(aug['image'])
            mask = aug['mask']
        
        if self.transform is None: img = Image.fromarray(img)
        
        #normalization
        if (self.black_white): t = T.Compose([T.ToTensor(), T.Normalize(self.mean, self.std), T.Grayscale()])
        else: t = T.Compose([T.ToTensor(), T.Normalize(self.mean, self.std)])
        
        img = t(img)
        mask = torch.from_numpy(mask).long()
            
        return img, mask
    
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
    
    def __init__(self, img_path, mask_path, rng_path, X, mean, std, transform=None, test = False, black_white=True):
        self.img_path = img_path
        self.mask_path = mask_path
        self.range_path = rng_path
        self.X = X
        self.transform = transform
        self.mean = mean
        self.std = std
        self.black_white = black_white
        self.test = test
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        
        #lettura immagini
        img = cv2.imread(self.img_path + self.X[idx] + '.JPG')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #from BGR to RGB
        if (self.test): mask = cv2.imread(self.mask_path + self.X[idx] + '_merged.png')
        else: mask = cv2.imread(self.mask_path + self.X[idx] + '.png')
        rng30 = cv2.imread(self.range_path + (self.X[idx]).replace("EDR","RNG") + '.png')
        
        #merging mask
        mask = self.merge_real_mars(mask,rng30)
        
        #transformations
        if self.transform is not None:
            aug = self.transform(image=img, mask=mask)
            img = Image.fromarray(aug['image'])
            mask = aug['mask']
        
        if self.transform is None: img = Image.fromarray(img)
        
        #normalization
        if (self.black_white): t = T.Compose([T.ToTensor(), T.Normalize(self.mean, self.std), T.Grayscale()])
        else: t = T.Compose([T.ToTensor(), T.Normalize(self.mean, self.std)])
        img = t(img)
        mask = torch.from_numpy(mask).long()
            
        return img, mask
    
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

# MarsDataset-v3
class MarsDatasetv3(Dataset):
    
    def __init__(self, img_path, mask_path, sky_path, X, mean, std, transform = None, black_white = True):
        self.img_path = img_path
        self.mask_path = mask_path
        self.sky_path = sky_path
        self.X = X
        self.transform = transform
        self.mean = mean
        self.std = std
        self.black_white = black_white
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        
        #lettura immagini
        img = cv2.imread(self.img_path + self.X[idx] + '.png')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #from BGR to RGB
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
        if self.transform is not None:
            aug = self.transform(image=img, mask=mask)
            img = Image.fromarray(aug['image'])
            mask = aug['mask']
        
        if self.transform is None: img = Image.fromarray(img)
        
        #normalization
        if (self.black_white): t = T.Compose([T.ToTensor(), T.Normalize(self.mean, self.std), T.Grayscale()])
        else: t = T.Compose([T.ToTensor(), T.Normalize(self.mean, self.std)])
        
        img = t(img)
        mask = torch.from_numpy(mask).long()
            
        return img, mask

    
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
    
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
