import numpy as np 
import pandas as pd
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

import time, math, random
import os
from tqdm import tqdm

from torchsummary import summary
import segmentation_models_pytorch as smp

IMAGE_PATH = "synthetic-moon-dataset/images/render/"
MASK_PATH = "synthetic-moon-dataset/images/ground/"

# Check whether the specified path exists or not
if (not os.path.exists("figs/")): os.makedirs("figs/")
if (not os.path.exists("models/")): os.makedirs("models/")

#check if cuda is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Training on Device: ", device)

#fixing random seed
random_seed = 42
torch.manual_seed(random_seed)
if (device != "cpu"):
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
random.seed(random_seed)
np.random.seed(random_seed) #if you use numpy

#loading input images and masks
black_white = True #transforms it into grayscale
if (black_white): color_channels = 1
else: color_channels = 3

#functions and classes
def create_df():
    name = []
    for dirname, _, filenames in os.walk(IMAGE_PATH):
        for filename in filenames:
            imname = filename.split('.')[0]
            imname = imname.replace('render','')
            name.append(imname)
    
    return pd.DataFrame({'id': name}, index = np.arange(0, len(name)))

def create_df_sorted():
    name = []
    for filename in sorted(os.listdir(IMAGE_PATH)): 
        imname = filename.split('.')[0]    
        imname = imname.replace('render','')
        name.append(imname)
    return pd.DataFrame({'id': name}, index = np.arange(0, len(name)))

def plotinput(idxplot,df):

    img = Image.open(IMAGE_PATH + 'render' + df['id'][idxplot] + '.png')
    mask = Image.open(MASK_PATH + 'ground' + df['id'][idxplot] + '.png')

    print('Image Size', np.asarray(img).shape)
    print('Mask Size', np.asarray(mask).shape)

    #note the different colors
    plt.figure()
    plt.imshow(img)
    plt.imshow(mask, alpha=0.6)
    plt.title('Picture with Mask Applied')
    plt.show()

#mean and standard deviations
def getmeanstd(X_train):

    n_train = len(X_train)
    #mean computation
    X_train_imgs_r = 0.
    X_train_imgs_g = 0.
    X_train_imgs_b = 0.

    #std dev computation
    X_train_imgs_r_sq = 0.
    X_train_imgs_g_sq = 0.
    X_train_imgs_b_sq = 0.

    n_tot = n_train*480*720

    for i_train,x in enumerate(X_train):
        img = cv2.imread(IMAGE_PATH + 'render' + x + '.png')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #from BGR to RGB
        img = np.array(img,dtype = np.int64)
        
        X_train_imgs_r += np.sum(img[:,:,0])
        X_train_imgs_g += np.sum(img[:,:,1])
        X_train_imgs_b += np.sum(img[:,:,2])
        
        X_train_imgs_r_sq += np.sum(np.square(img[:,:,0]))
        X_train_imgs_g_sq += np.sum(np.square(img[:,:,1]))
        X_train_imgs_b_sq += np.sum(np.square(img[:,:,2]))
        
    mean = [X_train_imgs_r/(n_tot),X_train_imgs_g/(n_tot),X_train_imgs_b/(n_tot)]

    std_dev = [X_train_imgs_r_sq/(n_tot) - (mean[0]*mean[0]),
            X_train_imgs_g_sq/(n_tot) - (mean[1]*mean[1]),
            X_train_imgs_b_sq/(n_tot) - (mean[2]*mean[2])]

    std_dev = np.sqrt(std_dev)

    mean = np.divide(mean,255).tolist()
    std_dev = np.divide(std_dev,255).tolist()
    return mean,std_dev

def getdistcts(train_set,val_set,test_set):

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
        
        if (i_test % (int(len(test_set)/10)) == 0): print("- %i / %i" % (i_test,len(test_set)))
        data = x[1]
        
        pix_test += data.shape[0]*data.shape[1]
        test_count[0] += np.count_nonzero(data == 0)
        test_count[1] += np.count_nonzero(data == 1)
        test_count[2] += np.count_nonzero(data == 2)

    test_count = np.divide(test_count,pix_test)

    print("\n- VALIDATION")
    for i_val,x in enumerate(val_set):
        
        if (i_val % (int(len(val_set)/10)) == 0): print("- %i / %i" % (i_val,len(val_set)))
        data = x[1]
        
        pix_val += data.shape[0]*data.shape[1]
        val_count[0] += np.count_nonzero(data == 0)
        val_count[1] += np.count_nonzero(data == 1)
        val_count[2] += np.count_nonzero(data == 2)
        
    val_count = np.divide(val_count,pix_val)
    return train_count, val_count, test_count

#modifying balanced accuracy
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

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def fit(epochs, model, train_loader, val_loader, criterion, optimizers, schedulers, metric_names, stopat = None):
    
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
    
    for e in range(epochs):

        since = time.time()
        running_loss = 0
        
        train_metr_epoch = [0,0,0]
        test_metr_epoch = [0,0,0]
        lr_list = [] #learning rate for each optimizer e.g. [0.001,0.005,0.01]
        
        #training loop
        model.train()
        
        for i, data in enumerate(tqdm(train_loader)):
            
            if (stopat is not None and i == stopat): break

            #training phase
            image, mask = data
            
            image = image.to(device)
            mask = mask.to(device)
            
            #forward
            output = model(image)
            loss = criterion(output, mask)
            
            #evaluation metrics
            for imetr,metr in enumerate(metric_names):
                train_metr_epoch[imetr] += pixel_metrics(output,mask,metr)
            
            #backward
            loss.backward()
            
            for optimizer in optimizers:
                
                optimizer.step() #update weight          
                optimizer.zero_grad() #reset gradient
            
                #step the learning rate
                if (i==0): lr_list.append(get_lr(optimizer)) #need it only for first element of dataloader
                    
            running_loss += loss.item()
            
        #end training 
        
        #step the learning rate
        lrs.append(lr_list)
        
        #scheduler step at each epoch
        for scheduler in schedulers:
            if (scheduler is not None): scheduler.step()
        
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
                loss = criterion(output, mask)  
                
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
        
            #saving every 5 decrease
            #if decrease % 5 == 0:
            #    print('Saving model...')
            #    torch.save(model, 'Unet-{decrease}.pt')

        if (test_loss/len(val_loader)) > min_loss:
            
            not_improve += 1
            min_loss = (test_loss/len(val_loader))
            print('Loss Not Decreasing for {}-th Time'.format(not_improve))
            
            if not_improve == 7:
                print('Loss not decreasing for 7 times, Stop Training')
                #break
                
        print("Epoch:{}/{}..".format(e+1, epochs),
              "Train Loss: {:.3f}..".format(running_loss/len(train_loader)),
              "Val Loss: {:.3f}..".format(test_loss/len(val_loader)))
        
        for imetr,metr in enumerate(metric_names):
            print("Train " + metr + ": {:.3f}".format(train_metrics[e][imetr]))
            print("Val " + metr + ": {:.3f}".format(test_metrics[e][imetr]))
            
    history = {'train_loss' : train_losses, 'val_loss': test_losses,
               'train_metric' : train_metrics, 'val_metric': test_metrics,
               'lrs': lrs}
    
    print('Total time: {:.2f} m' .format((time.time()- fit_time)/60))
    return history

def plot_loss(history):
    
    plt.figure()
    plt.plot(history['val_loss'], label='Validation Loss', marker='o', color = "blue", linestyle = "dashed")
    plt.plot(history['train_loss'], label='Training Loss', marker='o', color = "green",linestyle = "dashed")
    plt.title('Loss per epoch')
    plt.ylabel('Categorical Cross Entropy Loss')
    plt.xlabel('Epoch')
    plt.legend(loc = "best")
    plt.grid()
    plt.show()
    
def plot_score(history,metric_names):
    
    for imetr,metr in enumerate(metric_names):
        
        train_score = [history['train_metric'][e][imetr] for e in range(len(history['train_metric']))] 
        val_score = [history['val_metric'][e][imetr] for e in range(len(history['val_metric']))] 
        
        plt.figure()
        plt.plot(train_score, label = "Training " + metr, marker='o', color = "green", linestyle = "dashed")
        plt.plot(val_score, label = "Validation " + metr, marker='o', color = "blue",linestyle = "dashed")
        plt.title('Score per epoch')
        plt.ylabel("Mean " + metr)
        plt.xlabel('Epoch')
        plt.legend(loc = "best")
        plt.grid()
        plt.show()

def plot_lrs(history):
    
    nlrs = len(history['lrs'][0])
    lr_epoch = []
    colors = ["r","g","b","c","m","y"]
    labels = ["Encoder","Decoder","Segmentation Head"]
    
    for ilr in range(nlrs):
        lr_epoch.append([history['lrs'][e][ilr] for e in range(len(history['lrs']))])
        
    plt.figure()
    for ilr in range(nlrs):
        plt.plot(lr_epoch[ilr], marker='o', color = colors[ilr], label = labels[ilr], linestyle = "dashed")
    plt.title('Learning Rates per epoch')
    plt.ylabel("Learning Rate")
    plt.xlabel('Epoch')
    #plt.yscale("log")
    plt.legend(loc = "best")
    plt.grid()
    plt.show()

def evaluate(model,test_loader,metric_names):

    model.eval()
    model.to(device)
    confusion_matrix = np.zeros((3,3))
    test_metrics = [[],[],[]]

    with torch.no_grad():    
        
        for i, data in enumerate(tqdm(test_loader)):
            
            image, mask = data
            image = image.to(device)
            mask = mask.to(device)
            
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
            
            #output = output.cpu().squeeze(0)
    return test_metrics, confusion_matrix

#functions to compare results
def plot_loss_compare(histories,labels,lasts = None):
    
    colors = ["r","g","b","c","m","y"]
    
    plt.figure()
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
    plt.ylabel('Categorical Cross Entropy Loss');
    plt.xlabel('Epoch')
    plt.legend(loc = "best")
    plt.grid()
    plt.show()
    
def plot_score_compare(histories,labels,metric_names,lasts = None):
    
    colors = ["r","g","b","c","m","y"]
    
    for imetr,metr in enumerate(metric_names):
        
        plt.figure()
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
        plt.show()

def predicted_mask(model,image):
    
    image = image.to(device)
    output = model(image)
    
    #applying softmax and choosing best class with argmax
    output = F.softmax(output, dim=1) #on all colors
    output = torch.argmax(output, dim=1) #on all colors
    
    return output

#dataset
class FakeMoonDataset(Dataset):
    
    def __init__(self, img_path, mask_path, X, mean, std, transform=None):
        self.img_path = img_path
        self.mask_path = mask_path
        self.X = X
        self.transform = transform
        self.mean = mean
        self.std = std
        
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
        if (black_white): t = T.Compose([T.ToTensor(), T.Normalize(self.mean, self.std), T.Grayscale()])
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