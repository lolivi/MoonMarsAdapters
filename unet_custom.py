import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
import sys,os
import numpy as np
import psutil

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

#resnet18
class Unet_18(nn.Module):

    def __init__(self, in_channels, num_classes, adapters):

        super(Unet_18, self).__init__()
        self.adapters = adapters

        #----------------------------------------------------------------        
        #encoder layers (encoder_block_layernumber_typeoflayer)
        #----------------------------------------------------------------

        #first block
        self.encoder_1_1_conv = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)   
        self.encoder_1_2_batchnorm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
        self.encoder_1_3_relu = nn.ReLU(inplace=True)
        self.encoder_1_4_maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        #adapter
        if (self.adapters):
            self.adapter_1_1_batchnorm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_1_2_conv = nn.Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_1_2_conv)
            self.init_bnorm(self.adapter_1_1_batchnorm)

        #second block
        self.encoder_2_1_conv = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        #adapter
        if (self.adapters):
            self.adapter_2_1_batchnorm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_2_2_conv = nn.Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_2_2_conv)
            self.init_bnorm(self.adapter_2_1_batchnorm)

        self.encoder_2_2_batchnorm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.encoder_2_3_relu = nn.ReLU(inplace=True)
        self.encoder_2_4_conv = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        #adapter
        if (self.adapters):
            self.adapter_3_1_batchnorm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_3_2_conv = nn.Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_3_2_conv)
            self.init_bnorm(self.adapter_3_1_batchnorm)

        self.encoder_2_5_batchnorm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.encoder_2_6_relu = nn.ReLU(inplace=True)
        self.encoder_2_7_conv = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        #adapter
        if (self.adapters):
            self.adapter_4_1_batchnorm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_4_2_conv = nn.Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_4_2_conv)
            self.init_bnorm(self.adapter_4_1_batchnorm)

        self.encoder_2_8_batchnorm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.encoder_2_9_relu = nn.ReLU(inplace=True)
        self.encoder_2_10_conv = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        #adapter
        if (self.adapters):
            self.adapter_5_1_batchnorm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_5_2_conv = nn.Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_5_2_conv)
            self.init_bnorm(self.adapter_5_1_batchnorm)

        self.encoder_2_11_batchnorm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.encoder_2_12_relu = nn.ReLU(inplace=True)

        #third block
        self.encoder_3_1_conv = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

        #adapter
        if (self.adapters):
            self.adapter_6_1_batchnorm = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_6_2_conv = nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_6_2_conv)
            self.init_bnorm(self.adapter_6_1_batchnorm)

        self.encoder_3_2_batchnorm = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.encoder_3_3_relu = nn.ReLU(inplace=True)
        self.encoder_3_4_conv = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        #adapter
        if (self.adapters):
            self.adapter_7_1_batchnorm = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_7_2_conv = nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_7_2_conv)
            self.init_bnorm(self.adapter_7_1_batchnorm)

        self.encoder_3_5_batchnorm = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        #need it for concatenation
        self.encoder_3_6_downsample = nn.Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
        self.encoder_3_7_batchnorm = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.encoder_3_8_relu = nn.ReLU(inplace=True)

        self.encoder_3_9_conv = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        #adapter
        if (self.adapters):
            self.adapter_8_1_batchnorm = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_8_2_conv = nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_8_2_conv)
            self.init_bnorm(self.adapter_8_1_batchnorm)

        self.encoder_3_10_batchnorm = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.encoder_3_11_relu = nn.ReLU(inplace=True)
        self.encoder_3_12_conv = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        #adapter
        if (self.adapters):
            self.adapter_9_1_batchnorm = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_9_2_conv = nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_9_2_conv)
            self.init_bnorm(self.adapter_9_1_batchnorm)

        self.encoder_3_13_batchnorm = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.encoder_3_14_relu = nn.ReLU(inplace=True)

        #fourth block
        self.encoder_4_1_conv = nn.Conv2d(128,256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

        #adapter
        if (self.adapters):
            self.adapter_10_1_batchnorm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_10_2_conv = nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_10_2_conv)
            self.init_bnorm(self.adapter_10_1_batchnorm)

        self.encoder_4_2_batchnorm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.encoder_4_3_relu = nn.ReLU(inplace=True)
        self.encoder_4_4_conv = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        #adapter
        if (self.adapters):
            self.adapter_11_1_batchnorm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_11_2_conv = nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_11_2_conv)
            self.init_bnorm(self.adapter_11_1_batchnorm)

        self.encoder_4_5_batchnorm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        #need it for concatenation
        self.encoder_4_6_downsample = nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
        self.encoder_4_7_batchnorm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.encoder_4_8_relu = nn.ReLU(inplace=True)

        self.encoder_4_9_conv = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        #adapter
        if (self.adapters):
            self.adapter_12_1_batchnorm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_12_2_conv = nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_12_2_conv)
            self.init_bnorm(self.adapter_12_1_batchnorm)

        self.encoder_4_10_batchnorm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.encoder_4_11_relu = nn.ReLU(inplace=True)
        self.encoder_4_12_conv = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        #adapter
        if (self.adapters):
            self.adapter_13_1_batchnorm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_13_2_conv = nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_13_2_conv)
            self.init_bnorm(self.adapter_13_1_batchnorm)

        self.encoder_4_13_batchnorm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.encoder_4_14_relu = nn.ReLU(inplace=True)

        #fifth block
        self.encoder_5_1_conv = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

        #adapter
        if (self.adapters):
            self.adapter_14_1_batchnorm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_14_2_conv = nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_14_2_conv)
            self.init_bnorm(self.adapter_14_1_batchnorm)

        self.encoder_5_2_batchnorm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.encoder_5_3_relu = nn.ReLU(inplace=True)
        self.encoder_5_4_conv = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        #adapter
        if (self.adapters):
            self.adapter_15_1_batchnorm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_15_2_conv = nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_15_2_conv)
            self.init_bnorm(self.adapter_15_1_batchnorm)

        self.encoder_5_5_batchnorm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        #need it for concatenation
        self.encoder_5_6_downsample = nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
        self.encoder_5_7_batchnorm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.encoder_5_8_relu = nn.ReLU(inplace=True)

        self.encoder_5_9_conv = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        #adapter
        if (self.adapters):
            self.adapter_16_1_batchnorm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_16_2_conv = nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_16_2_conv)
            self.init_bnorm(self.adapter_16_1_batchnorm)

        self.encoder_5_10_batchnorm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.encoder_5_11_relu = nn.ReLU(inplace=True)
        self.encoder_5_12_conv = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        #adapter
        if (self.adapters):
            self.adapter_17_1_batchnorm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_17_2_conv = nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_17_2_conv)
            self.init_bnorm(self.adapter_17_1_batchnorm)

        self.encoder_5_13_batchnorm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.encoder_5_14_relu = nn.ReLU(inplace=True)

        #----------------------------------------------------------------        
        #decoder layers (decoder_block_layernumber_typeoflayer)
        #----------------------------------------------------------------

        #first decoder block
        self.decoder_1_1_conv = nn.Conv2d(768, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        #adapter
        if (self.adapters):
            self.adapter_18_1_batchnorm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_18_2_conv = nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_18_2_conv)
            self.init_bnorm(self.adapter_18_1_batchnorm)

        self.decoder_1_2_batchnorm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.decoder_1_3_relu = nn.ReLU(inplace=True)
        self.decoder_1_4_conv = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        
        #adapter
        if (self.adapters):
            self.adapter_19_1_batchnorm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_19_2_conv = nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_19_2_conv)
            self.init_bnorm(self.adapter_19_1_batchnorm)
        
        self.decoder_1_5_batchnorm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.decoder_1_6_relu = nn.ReLU(inplace=True)

        #second decoder block
        self.decoder_2_1_conv = nn.Conv2d(384, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        #adapter
        if (self.adapters):
            self.adapter_20_1_batchnorm = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_20_2_conv = nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_20_2_conv)
            self.init_bnorm(self.adapter_20_1_batchnorm)

        self.decoder_2_2_batchnorm = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.decoder_2_3_relu = nn.ReLU(inplace=True)
        self.decoder_2_4_conv = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        #adapter
        if (self.adapters):
            self.adapter_21_1_batchnorm = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_21_2_conv = nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_21_2_conv)
            self.init_bnorm(self.adapter_21_1_batchnorm)

        self.decoder_2_5_batchnorm = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.decoder_2_6_relu = nn.ReLU(inplace=True)

        #third decoder block
        self.decoder_3_1_conv = nn.Conv2d(192, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        #adapter
        if (self.adapters):
            self.adapter_22_1_batchnorm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_22_2_conv = nn.Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_22_2_conv)
            self.init_bnorm(self.adapter_22_1_batchnorm)

        self.decoder_3_2_batchnorm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.decoder_3_3_relu = nn.ReLU(inplace=True)
        self.decoder_3_4_conv = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        #adapter
        if (self.adapters):
            self.adapter_23_1_batchnorm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_23_2_conv = nn.Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_23_2_conv)
            self.init_bnorm(self.adapter_23_1_batchnorm)

        self.decoder_3_5_batchnorm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.decoder_3_6_relu = nn.ReLU(inplace=True)

        #fourth decoder block
        self.decoder_4_1_conv = nn.Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        #adapter
        if (self.adapters):
            self.adapter_24_1_batchnorm = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_24_2_conv = nn.Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_24_2_conv)
            self.init_bnorm(self.adapter_24_1_batchnorm)

        self.decoder_4_2_batchnorm = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.decoder_4_3_relu = nn.ReLU(inplace=True)
        self.decoder_4_4_conv = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        #adapter
        if (self.adapters):
            self.adapter_25_1_batchnorm = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_25_2_conv = nn.Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_25_2_conv)
            self.init_bnorm(self.adapter_25_1_batchnorm)

        self.decoder_4_5_batchnorm = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.decoder_4_6_relu = nn.ReLU(inplace=True)

        #fifth decoder block
        self.decoder_5_1_conv = nn.Conv2d(32, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        #adapter
        if (self.adapters):
            self.adapter_26_1_batchnorm = nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_26_2_conv = nn.Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_26_2_conv)
            self.init_bnorm(self.adapter_26_1_batchnorm)

        self.decoder_5_2_batchnorm = nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.decoder_5_3_relu = nn.ReLU(inplace=True)
        self.decoder_5_4_conv = nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        #adapter
        if (self.adapters):
            self.adapter_27_1_batchnorm = nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_27_2_conv = nn.Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_27_2_conv)
            self.init_bnorm(self.adapter_27_1_batchnorm)

        self.decoder_5_5_batchnorm = nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.decoder_5_6_relu = nn.ReLU(inplace=True)

        #final segmentation head
        self.decoder_6_1_conv = nn.Conv2d(16, num_classes, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    def init_convweights(self, layer):
        layer.weight.data.zero_()
        return

    def init_bnorm(self, layer):
        layer.weight.data.fill_(1)
        layer.bias.data.zero_()
        layer.running_mean.zero_()
        layer.running_var.fill_(1)
        layer.num_batches_tracked.zero_()
        return 
    
    def forward(self, x):

        # print("Input Shape:")
        # print(x.size())

        #first block
        block1 = self.encoder_1_1_conv(x)

        if (self.adapters):
            adapt1 = self.adapter_1_1_batchnorm(block1)
            adapt1 = self.adapter_1_2_conv(adapt1)
            block1 = block1 + adapt1

        block1 = self.encoder_1_2_batchnorm(block1)
        block1 = self.encoder_1_3_relu(block1)
        x1 = block1 #save it for decoder
        block1 = self.encoder_1_4_maxpool(block1)

        #second block
        block2 = self.encoder_2_1_conv(block1)

        if (self.adapters):
            adapt2 = self.adapter_2_1_batchnorm(block2)
            adapt2 = self.adapter_2_2_conv(adapt2)
            block2 = block2 + adapt2

        block2 = self.encoder_2_2_batchnorm(block2) 
        block2 = self.encoder_2_3_relu(block2)
        block2 = self.encoder_2_4_conv(block2)

        if (self.adapters):
            adapt3 = self.adapter_3_1_batchnorm(block2)
            adapt3 = self.adapter_3_2_conv(adapt3)
            block2 = block2 + adapt3

        block2 = self.encoder_2_5_batchnorm(block2)

        #skip connection
        x2 = block1 + block2
        x2 = self.encoder_2_6_relu(x2)

        block2 = self.encoder_2_7_conv(x2)

        if (self.adapters):
            adapt4 = self.adapter_4_1_batchnorm(block2)
            adapt4 = self.adapter_4_2_conv(adapt4)
            block2 = block2 + adapt4

        block2 = self.encoder_2_8_batchnorm(block2) 
        block2 = self.encoder_2_9_relu(block2)
        block2 = self.encoder_2_10_conv(block2)

        if (self.adapters):
            adapt5 = self.adapter_5_1_batchnorm(block2)
            adapt5 = self.adapter_5_2_conv(adapt5)
            block2 = block2 + adapt5

        block2 = self.encoder_2_11_batchnorm(block2)

        #skip connection
        x3 = x2 + block2
        x3 = self.encoder_2_12_relu(x3)

        #third block
        block3 = self.encoder_3_1_conv(x3)

        if (self.adapters):
            adapt6 = self.adapter_6_1_batchnorm(block3)
            adapt6 = self.adapter_6_2_conv(adapt6)
            block3 = block3 + adapt6

        block3 = self.encoder_3_2_batchnorm(block3) 
        block3 = self.encoder_3_3_relu(block3) 
        block3 = self.encoder_3_4_conv(block3) 

        if (self.adapters):
            adapt7 = self.adapter_7_1_batchnorm(block3)
            adapt7 = self.adapter_7_2_conv(adapt7)
            block3 = block3 + adapt7

        block3 = self.encoder_3_5_batchnorm(block3) 

        #need it for concatenation
        x4 = self.encoder_3_6_downsample(x3) 
        x4 = self.encoder_3_7_batchnorm(x4)
        x4 = x4 + block3
        x4 = self.encoder_3_8_relu(x4) 

        block3 = self.encoder_3_9_conv(x4) 

        if (self.adapters):
            adapt8 = self.adapter_8_1_batchnorm(block3)
            adapt8 = self.adapter_8_2_conv(adapt8)
            block3 = block3 + adapt8

        block3 = self.encoder_3_10_batchnorm(block3)
        block3 = self.encoder_3_11_relu(block3)
        block3 = self.encoder_3_12_conv(block3)

        if (self.adapters):
            adapt9 = self.adapter_9_1_batchnorm(block3)
            adapt9 = self.adapter_9_2_conv(adapt9)
            block3 = block3 + adapt9

        block3 = self.encoder_3_13_batchnorm(block3)

        #skip connection
        x5 = x4 + block3
        x5 = self.encoder_3_14_relu(x5)

        #fourth block
        block4 = self.encoder_4_1_conv(x5)

        if (self.adapters):
            adapt10 = self.adapter_10_1_batchnorm(block4)
            adapt10 = self.adapter_10_2_conv(adapt10)
            block4 = block4 + adapt10

        block4 = self.encoder_4_2_batchnorm(block4) 
        block4 = self.encoder_4_3_relu(block4) 
        block4 = self.encoder_4_4_conv(block4) 

        if (self.adapters):
            adapt11 = self.adapter_11_1_batchnorm(block4)
            adapt11 = self.adapter_11_2_conv(adapt11)
            block4 = block4 + adapt11

        block4 = self.encoder_4_5_batchnorm(block4) 

        #need it for concatenation
        x6 = self.encoder_4_6_downsample(x5) 
        x6 = self.encoder_4_7_batchnorm(x6)
        x6 = x6 + block4
        x6 = self.encoder_4_8_relu(x6)

        block4 = self.encoder_4_9_conv(x6) 

        if (self.adapters):
            adapt12 = self.adapter_12_1_batchnorm(block4)
            adapt12 = self.adapter_12_2_conv(adapt12)
            block4 = block4 + adapt12

        block4 = self.encoder_4_10_batchnorm(block4)
        block4 = self.encoder_4_11_relu(block4)
        block4 = self.encoder_4_12_conv(block4)

        if (self.adapters):
            adapt13 = self.adapter_13_1_batchnorm(block4)
            adapt13 = self.adapter_13_2_conv(adapt13)
            block4 = block4 + adapt13

        block4 = self.encoder_4_13_batchnorm(block4)
        
        #skip connection
        x7 = x6 + block4
        x7 = self.encoder_4_14_relu(x7)

        #fifth block
        block5 = self.encoder_5_1_conv(x7)

        if (self.adapters):
            adapt14 = self.adapter_14_1_batchnorm(block5)
            adapt14 = self.adapter_14_2_conv(adapt14)
            block5 = block5 + adapt14

        block5 = self.encoder_5_2_batchnorm(block5) 
        block5 = self.encoder_5_3_relu(block5) 
        block5 = self.encoder_5_4_conv(block5) 

        if (self.adapters):
            adapt15 = self.adapter_15_1_batchnorm(block5)
            adapt15 = self.adapter_15_2_conv(adapt15)
            block5 = block5 + adapt15

        block5 = self.encoder_5_5_batchnorm(block5) 

        #need it for concatenation
        x8 = self.encoder_5_6_downsample(x7) 
        x8 = self.encoder_5_7_batchnorm(x8)
        x8 = x8 + block5
        x8 = self.encoder_5_8_relu(x8)

        block5 = self.encoder_5_9_conv(x8) 

        if (self.adapters):
            adapt16 = self.adapter_16_1_batchnorm(block5)
            adapt16 = self.adapter_16_2_conv(adapt16)
            block5 = block5 + adapt16

        block5 = self.encoder_5_10_batchnorm(block5)
        block5 = self.encoder_5_11_relu(block5)
        block5 = self.encoder_5_12_conv(block5)

        if (self.adapters):
            adapt17 = self.adapter_17_1_batchnorm(block5)
            adapt17 = self.adapter_17_2_conv(adapt17)
            block5 = block5 + adapt17

        block5 = self.encoder_5_13_batchnorm(block5)

        #skip connection
        x9 = x8 + block5 #bottleneck
        x9 = self.encoder_5_14_relu(x9)

        # print("Encoder Shapes: ")
        # print(x1.size(),block2.size(),block3.size(),block4.size(),block5.size())

        #upsampling + concatenation
        y1 = F.interpolate(x9, scale_factor=2, mode="nearest")
        y1 = torch.cat([y1,x7],dim=1) #512 + 256

        #first decoder block
        deblock1 = self.decoder_1_1_conv(y1)

        if (self.adapters):
            adapt18 = self.adapter_18_1_batchnorm(deblock1)
            adapt18 = self.adapter_18_2_conv(adapt18)
            deblock1 = deblock1 + adapt18

        deblock1 = self.decoder_1_2_batchnorm(deblock1)
        deblock1 = self.decoder_1_3_relu(deblock1)
        deblock1 = self.decoder_1_4_conv(deblock1)

        if (self.adapters):
            adapt19 = self.adapter_19_1_batchnorm(deblock1)
            adapt19 = self.adapter_19_2_conv(adapt19)
            deblock1 = deblock1 + adapt19

        deblock1 = self.decoder_1_5_batchnorm(deblock1)
        deblock1 = self.decoder_1_6_relu(deblock1)

        #upsampling + concatenation
        deblock1 = F.interpolate(deblock1, scale_factor=2, mode="nearest")
        y2 = torch.cat([deblock1,x5],dim=1) #256 + 128

        #second decoder block
        deblock2 = self.decoder_2_1_conv(y2)

        if (self.adapters):
            adapt20 = self.adapter_20_1_batchnorm(deblock2)
            adapt20 = self.adapter_20_2_conv(adapt20)
            deblock2 = deblock2 + adapt20

        deblock2 = self.decoder_2_2_batchnorm(deblock2)
        deblock2 = self.decoder_2_3_relu(deblock2)
        deblock2 = self.decoder_2_4_conv(deblock2)

        if (self.adapters):
            adapt21 = self.adapter_21_1_batchnorm(deblock2)
            adapt21 = self.adapter_21_2_conv(adapt21)
            deblock2 = deblock2 + adapt21

        deblock2 = self.decoder_2_5_batchnorm(deblock2)
        deblock2 = self.decoder_2_6_relu(deblock2)

        #upsampling + concatenation
        deblock2 = F.interpolate(deblock2, scale_factor=2, mode="nearest")
        y3 = torch.cat([deblock2,x3],dim=1) #128 + 64

        #third decoder block
        deblock3 = self.decoder_3_1_conv(y3)

        if (self.adapters):
            adapt22 = self.adapter_22_1_batchnorm(deblock3)
            adapt22 = self.adapter_22_2_conv(adapt22)
            deblock3 = deblock3 + adapt22

        deblock3 = self.decoder_3_2_batchnorm(deblock3)
        deblock3 = self.decoder_3_3_relu(deblock3)
        deblock3 = self.decoder_3_4_conv(deblock3)

        if (self.adapters):
            adapt23 = self.adapter_23_1_batchnorm(deblock3)
            adapt23 = self.adapter_23_2_conv(adapt23)
            deblock3 = deblock3 + adapt23

        deblock3 = self.decoder_3_5_batchnorm(deblock3)
        deblock3 = self.decoder_3_6_relu(deblock3)

        #upsampling + concatenation
        deblock3 = F.interpolate(deblock3, scale_factor=2, mode="nearest")
        y4 = torch.cat([deblock3,x1],dim=1) #64 + 64

        #fourth decoder block
        deblock4 = self.decoder_4_1_conv(y4)

        if (self.adapters):
            adapt24 = self.adapter_24_1_batchnorm(deblock4)
            adapt24 = self.adapter_24_2_conv(adapt24)
            deblock4 = deblock4 + adapt24

        deblock4 = self.decoder_4_2_batchnorm(deblock4)
        deblock4 = self.decoder_4_3_relu(deblock4)
        deblock4 = self.decoder_4_4_conv(deblock4)

        if (self.adapters):
            adapt25 = self.adapter_25_1_batchnorm(deblock4)
            adapt25 = self.adapter_25_2_conv(adapt25)
            deblock4 = deblock4 + adapt25

        deblock4 = self.decoder_4_5_batchnorm(deblock4)
        deblock4 = self.decoder_4_6_relu(deblock4)

        #no concatenation
        deblock4 = F.interpolate(deblock4, scale_factor=2, mode="nearest")
        y5 = deblock4 #32

        #fifth decoder block (segmentation head)
        deblock5 = self.decoder_5_1_conv(y5)

        if (self.adapters):
            adapt26 = self.adapter_26_1_batchnorm(deblock5)
            adapt26 = self.adapter_26_2_conv(adapt26)
            deblock5 = deblock5 + adapt26

        deblock5 = self.decoder_5_2_batchnorm(deblock5)
        deblock5 = self.decoder_5_3_relu(deblock5)
        deblock5 = self.decoder_5_4_conv(deblock5)

        if (self.adapters):
            adapt27 = self.adapter_27_1_batchnorm(deblock5)
            adapt27 = self.adapter_27_2_conv(adapt27)
            deblock5 = deblock5 + adapt27

        deblock5 = self.decoder_5_5_batchnorm(deblock5)
        deblock5 = self.decoder_5_6_relu(deblock5)

        #no concatenation (no upsampling in seg head)
        #deblock5 = F.interpolate(deblock5, scale_factor=2, mode="nearest")
        y6 = deblock5 #16

        #sixth decoder block
        deblock6 = self.decoder_6_1_conv(y6)

        # print("Decoder Shapes:")
        # print(y1.size(),y2.size(),y3.size(),y4.size(),y5.size(),y6.size(),deblock6.size())

        return deblock6

#resnet18
class Unet_18_ranked(nn.Module):

    def __init__(self, in_channels, num_classes, adapters):

        super(Unet_18_ranked, self).__init__()
        self.adapters = adapters

        #----------------------------------------------------------------        
        #encoder layers (encoder_block_layernumber_typeoflayer)
        #----------------------------------------------------------------

        #first block
        self.encoder_1_1_conv = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)   
        self.encoder_1_2_batchnorm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
        self.encoder_1_3_relu = nn.ReLU(inplace=True)
        self.encoder_1_4_maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        #adapter
        if (self.adapters):
            self.adapter_1_1_batchnorm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_1_2_conv = nn.Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_1_2_conv)
            self.init_bnorm(self.adapter_1_1_batchnorm)

        #second block
        self.encoder_2_1_conv = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        #adapter
        if (self.adapters):
            self.adapter_2_1_batchnorm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_2_2_conv = nn.Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_2_2_conv)
            self.init_bnorm(self.adapter_2_1_batchnorm)

        self.encoder_2_2_batchnorm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.encoder_2_3_relu = nn.ReLU(inplace=True)
        self.encoder_2_4_conv = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        #adapter
        if (self.adapters):
            self.adapter_3_1_batchnorm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_3_2_conv = nn.Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_3_2_conv)
            self.init_bnorm(self.adapter_3_1_batchnorm)

        self.encoder_2_5_batchnorm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.encoder_2_6_relu = nn.ReLU(inplace=True)
        self.encoder_2_7_conv = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        #adapter
        if (self.adapters):
            self.adapter_4_1_batchnorm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_4_2_conv = nn.Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_4_2_conv)
            self.init_bnorm(self.adapter_4_1_batchnorm)

        self.encoder_2_8_batchnorm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.encoder_2_9_relu = nn.ReLU(inplace=True)
        self.encoder_2_10_conv = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        #adapter
        if (self.adapters):
            self.adapter_5_1_batchnorm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_5_2_conv = nn.Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_5_2_conv)
            self.init_bnorm(self.adapter_5_1_batchnorm)

        self.encoder_2_11_batchnorm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.encoder_2_12_relu = nn.ReLU(inplace=True)

        #third block
        self.encoder_3_1_conv = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

        #adapter
        if (self.adapters):
            self.adapter_6_1_batchnorm = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_6_2_conv = nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_6_2_conv)
            self.init_bnorm(self.adapter_6_1_batchnorm)

        self.encoder_3_2_batchnorm = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.encoder_3_3_relu = nn.ReLU(inplace=True)
        self.encoder_3_4_conv = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        #adapter
        if (self.adapters):
            self.adapter_7_1_batchnorm = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_7_2_conv = nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_7_2_conv)
            self.init_bnorm(self.adapter_7_1_batchnorm)

        self.encoder_3_5_batchnorm = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        #need it for concatenation
        self.encoder_3_6_downsample = nn.Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
        self.encoder_3_7_batchnorm = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.encoder_3_8_relu = nn.ReLU(inplace=True)

        self.encoder_3_9_conv = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        #adapter
        if (self.adapters):
            self.adapter_8_1_batchnorm = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_8_2_conv = nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_8_2_conv)
            self.init_bnorm(self.adapter_8_1_batchnorm)

        self.encoder_3_10_batchnorm = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.encoder_3_11_relu = nn.ReLU(inplace=True)
        self.encoder_3_12_conv = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        #adapter
        if (self.adapters):
            self.adapter_9_1_batchnorm = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_9_2_conv = nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_9_2_conv)
            self.init_bnorm(self.adapter_9_1_batchnorm)

        self.encoder_3_13_batchnorm = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.encoder_3_14_relu = nn.ReLU(inplace=True)

        #fourth block
        self.encoder_4_1_conv = nn.Conv2d(128,256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

        #adapter
        if (self.adapters):
            self.adapter_10_1_batchnorm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_10_2_conv = nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_10_2_conv)
            self.init_bnorm(self.adapter_10_1_batchnorm)

        self.encoder_4_2_batchnorm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.encoder_4_3_relu = nn.ReLU(inplace=True)
        self.encoder_4_4_conv = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        #adapter
        if (self.adapters):
            self.adapter_11_1_batchnorm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_11_2_conv = nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_11_2_conv)
            self.init_bnorm(self.adapter_11_1_batchnorm)

        self.encoder_4_5_batchnorm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        #need it for concatenation
        self.encoder_4_6_downsample = nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
        self.encoder_4_7_batchnorm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.encoder_4_8_relu = nn.ReLU(inplace=True)

        self.encoder_4_9_conv = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        #adapter
        if (self.adapters):
            self.adapter_12_1_batchnorm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_12_2_conv = nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_12_2_conv)
            self.init_bnorm(self.adapter_12_1_batchnorm)

        self.encoder_4_10_batchnorm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.encoder_4_11_relu = nn.ReLU(inplace=True)
        self.encoder_4_12_conv = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        #adapter
        if (self.adapters):
            self.adapter_13_1_batchnorm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_13_2_conv = nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_13_2_conv)
            self.init_bnorm(self.adapter_13_1_batchnorm)

        self.encoder_4_13_batchnorm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.encoder_4_14_relu = nn.ReLU(inplace=True)

        #fifth block
        self.encoder_5_1_conv = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

        #adapter
        if (self.adapters):
            self.adapter_14_1_batchnorm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_14_2_conv = nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_14_2_conv)
            self.init_bnorm(self.adapter_14_1_batchnorm)

        self.encoder_5_2_batchnorm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.encoder_5_3_relu = nn.ReLU(inplace=True)
        self.encoder_5_4_conv = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        #adapter
        if (self.adapters):
            self.adapter_15_1_batchnorm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_15_2_conv = nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_15_2_conv)
            self.init_bnorm(self.adapter_15_1_batchnorm)

        self.encoder_5_5_batchnorm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        #need it for concatenation
        self.encoder_5_6_downsample = nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
        self.encoder_5_7_batchnorm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.encoder_5_8_relu = nn.ReLU(inplace=True)

        self.encoder_5_9_conv = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        #adapter
        if (self.adapters):
            self.adapter_16_1_batchnorm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_16_2_conv = nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_16_2_conv)
            self.init_bnorm(self.adapter_16_1_batchnorm)

        self.encoder_5_10_batchnorm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.encoder_5_11_relu = nn.ReLU(inplace=True)
        self.encoder_5_12_conv = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        #adapter
        if (self.adapters):
            self.adapter_17_1_batchnorm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_17_2_conv = nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_17_2_conv)
            self.init_bnorm(self.adapter_17_1_batchnorm)

        self.encoder_5_13_batchnorm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.encoder_5_14_relu = nn.ReLU(inplace=True)

        #----------------------------------------------------------------        
        #decoder layers (decoder_block_layernumber_typeoflayer)
        #----------------------------------------------------------------

        #first decoder block
        self.decoder_1_1_conv = nn.Conv2d(768, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        #adapter
        if (self.adapters):
            self.adapter_18_1_batchnorm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_18_2_conv = nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_18_2_conv)
            self.init_bnorm(self.adapter_18_1_batchnorm)

        self.decoder_1_2_batchnorm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.decoder_1_3_relu = nn.ReLU(inplace=True)
        self.decoder_1_4_conv = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        
        #adapter
        if (self.adapters):
            self.adapter_19_1_batchnorm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_19_2_conv = nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_19_2_conv)
            self.init_bnorm(self.adapter_19_1_batchnorm)
        
        self.decoder_1_5_batchnorm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.decoder_1_6_relu = nn.ReLU(inplace=True)

        #second decoder block
        self.decoder_2_1_conv = nn.Conv2d(384, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        #adapter
        if (self.adapters):
            self.adapter_20_1_batchnorm = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_20_2_conv = nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_20_2_conv)
            self.init_bnorm(self.adapter_20_1_batchnorm)

        self.decoder_2_2_batchnorm = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.decoder_2_3_relu = nn.ReLU(inplace=True)
        self.decoder_2_4_conv = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        #adapter
        if (self.adapters):
            self.adapter_21_1_batchnorm = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_21_2_conv = nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_21_2_conv)
            self.init_bnorm(self.adapter_21_1_batchnorm)

        self.decoder_2_5_batchnorm = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.decoder_2_6_relu = nn.ReLU(inplace=True)

        #third decoder block
        self.decoder_3_1_conv = nn.Conv2d(192, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        #adapter
        if (self.adapters):
            self.adapter_22_1_batchnorm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_22_2_conv = nn.Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_22_2_conv)
            self.init_bnorm(self.adapter_22_1_batchnorm)

        self.decoder_3_2_batchnorm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.decoder_3_3_relu = nn.ReLU(inplace=True)
        self.decoder_3_4_conv = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        #adapter
        if (self.adapters):
            self.adapter_23_1_batchnorm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_23_2_conv = nn.Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_23_2_conv)
            self.init_bnorm(self.adapter_23_1_batchnorm)

        self.decoder_3_5_batchnorm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.decoder_3_6_relu = nn.ReLU(inplace=True)

        #fourth decoder block
        self.decoder_4_1_conv = nn.Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        #adapter
        if (self.adapters):
            self.adapter_24_1_batchnorm = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_24_2_conv = nn.Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_24_2_conv)
            self.init_bnorm(self.adapter_24_1_batchnorm)

        self.decoder_4_2_batchnorm = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.decoder_4_3_relu = nn.ReLU(inplace=True)
        self.decoder_4_4_conv = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        #adapter
        if (self.adapters):
            self.adapter_25_1_batchnorm = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_25_2_conv = nn.Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_25_2_conv)
            self.init_bnorm(self.adapter_25_1_batchnorm)

        self.decoder_4_5_batchnorm = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.decoder_4_6_relu = nn.ReLU(inplace=True)

        #fifth decoder block
        self.decoder_5_1_conv = nn.Conv2d(32, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        #adapter
        if (self.adapters):
            self.adapter_26_1_batchnorm = nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_26_2_conv = nn.Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_26_2_conv)
            self.init_bnorm(self.adapter_26_1_batchnorm)

        self.decoder_5_2_batchnorm = nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.decoder_5_3_relu = nn.ReLU(inplace=True)
        self.decoder_5_4_conv = nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        #adapter
        if (self.adapters):
            self.adapter_27_1_batchnorm = nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_27_2_conv = nn.Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_27_2_conv)
            self.init_bnorm(self.adapter_27_1_batchnorm)

        self.decoder_5_5_batchnorm = nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.decoder_5_6_relu = nn.ReLU(inplace=True)

        #final segmentation head
        self.decoder_6_1_conv = nn.Conv2d(16, num_classes, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    def init_convweights(self, layer):
        layer.weight.data.zero_()
        return

    def init_bnorm(self, layer):
        layer.weight.data.fill_(1)
        layer.bias.data.zero_()
        layer.running_mean.zero_()
        layer.running_var.fill_(1)
        layer.num_batches_tracked.zero_()
        return 
    
    def forward(self, x):

        # print("Input Shape:")
        # print(x.size())

        #first block
        block1 = self.encoder_1_1_conv(x)

        if (self.adapters):
            adapt1 = self.adapter_1_1_batchnorm(block1)
            adapt1 = self.adapter_1_2_conv(adapt1)
            block1 = block1 + adapt1

        block1 = self.encoder_1_2_batchnorm(block1)
        block1 = self.encoder_1_3_relu(block1)
        x1 = block1 #save it for decoder
        block1 = self.encoder_1_4_maxpool(block1)

        #second block
        block2 = self.encoder_2_1_conv(block1)

        if (self.adapters):
            adapt2 = self.adapter_2_1_batchnorm(block2)
            adapt2 = self.adapter_2_2_conv(adapt2)
            block2 = block2 + adapt2

        block2 = self.encoder_2_2_batchnorm(block2) 
        block2 = self.encoder_2_3_relu(block2)
        block2 = self.encoder_2_4_conv(block2)

        if (self.adapters):
            adapt3 = self.adapter_3_1_batchnorm(block2)
            adapt3 = self.adapter_3_2_conv(adapt3)
            block2 = block2 + adapt3

        block2 = self.encoder_2_5_batchnorm(block2)

        #skip connection
        x2 = block1 + block2
        x2 = self.encoder_2_6_relu(x2)

        block2 = self.encoder_2_7_conv(x2)

        if (self.adapters):
            adapt4 = self.adapter_4_1_batchnorm(block2)
            adapt4 = self.adapter_4_2_conv(adapt4)
            block2 = block2 + adapt4

        block2 = self.encoder_2_8_batchnorm(block2) 
        block2 = self.encoder_2_9_relu(block2)
        block2 = self.encoder_2_10_conv(block2)

        if (self.adapters):
            adapt5 = self.adapter_5_1_batchnorm(block2)
            adapt5 = self.adapter_5_2_conv(adapt5)
            block2 = block2 + adapt5

        block2 = self.encoder_2_11_batchnorm(block2)

        #skip connection
        x3 = x2 + block2
        x3 = self.encoder_2_12_relu(x3)

        #third block
        block3 = self.encoder_3_1_conv(x3)

        if (self.adapters):
            adapt6 = self.adapter_6_1_batchnorm(block3)
            adapt6 = self.adapter_6_2_conv(adapt6)
            block3 = block3 + adapt6

        block3 = self.encoder_3_2_batchnorm(block3) 
        block3 = self.encoder_3_3_relu(block3) 
        block3 = self.encoder_3_4_conv(block3) 

        if (self.adapters):
            adapt7 = self.adapter_7_1_batchnorm(block3)
            adapt7 = self.adapter_7_2_conv(adapt7)
            block3 = block3 + adapt7

        block3 = self.encoder_3_5_batchnorm(block3) 

        #need it for concatenation
        x4 = self.encoder_3_6_downsample(x3) 
        x4 = self.encoder_3_7_batchnorm(x4)
        x4 = x4 + block3
        x4 = self.encoder_3_8_relu(x4) 

        block3 = self.encoder_3_9_conv(x4) 

        if (self.adapters):
            adapt8 = self.adapter_8_1_batchnorm(block3)
            adapt8 = self.adapter_8_2_conv(adapt8)
            block3 = block3 + adapt8

        block3 = self.encoder_3_10_batchnorm(block3)
        block3 = self.encoder_3_11_relu(block3)
        block3 = self.encoder_3_12_conv(block3)

        if (self.adapters):
            adapt9 = self.adapter_9_1_batchnorm(block3)
            adapt9 = self.adapter_9_2_conv(adapt9)
            block3 = block3 + adapt9

        block3 = self.encoder_3_13_batchnorm(block3)

        #skip connection
        x5 = x4 + block3
        x5 = self.encoder_3_14_relu(x5)

        #fourth block
        block4 = self.encoder_4_1_conv(x5)
        block4 = self.encoder_4_2_batchnorm(block4) 
        block4 = self.encoder_4_3_relu(block4) 
        block4 = self.encoder_4_4_conv(block4) 
        block4 = self.encoder_4_5_batchnorm(block4) 

        #need it for concatenation
        x6 = self.encoder_4_6_downsample(x5) 
        x6 = self.encoder_4_7_batchnorm(x6)
        x6 = x6 + block4
        x6 = self.encoder_4_8_relu(x6)

        block4 = self.encoder_4_9_conv(x6) 
        block4 = self.encoder_4_10_batchnorm(block4)
        block4 = self.encoder_4_11_relu(block4)
        block4 = self.encoder_4_12_conv(block4)
        block4 = self.encoder_4_13_batchnorm(block4)
        
        #skip connection
        x7 = x6 + block4
        x7 = self.encoder_4_14_relu(x7)

        #fifth block
        block5 = self.encoder_5_1_conv(x7)
        block5 = self.encoder_5_2_batchnorm(block5) 
        block5 = self.encoder_5_3_relu(block5) 
        block5 = self.encoder_5_4_conv(block5) 
        block5 = self.encoder_5_5_batchnorm(block5) 

        #need it for concatenation
        x8 = self.encoder_5_6_downsample(x7) 
        x8 = self.encoder_5_7_batchnorm(x8)
        x8 = x8 + block5
        x8 = self.encoder_5_8_relu(x8)

        block5 = self.encoder_5_9_conv(x8) 
        block5 = self.encoder_5_10_batchnorm(block5)
        block5 = self.encoder_5_11_relu(block5)
        block5 = self.encoder_5_12_conv(block5)
        block5 = self.encoder_5_13_batchnorm(block5)

        #skip connection
        x9 = x8 + block5 #bottleneck
        x9 = self.encoder_5_14_relu(x9)

        # print("Encoder Shapes: ")
        # print(x1.size(),block2.size(),block3.size(),block4.size(),block5.size())

        #upsampling + concatenation
        y1 = F.interpolate(x9, scale_factor=2, mode="nearest")
        y1 = torch.cat([y1,x7],dim=1) #512 + 256

        #first decoder block
        deblock1 = self.decoder_1_1_conv(y1)

        if (self.adapters):
            adapt18 = self.adapter_18_1_batchnorm(deblock1)
            adapt18 = self.adapter_18_2_conv(adapt18)
            deblock1 = deblock1 + adapt18

        deblock1 = self.decoder_1_2_batchnorm(deblock1)
        deblock1 = self.decoder_1_3_relu(deblock1)
        deblock1 = self.decoder_1_4_conv(deblock1)
        deblock1 = self.decoder_1_5_batchnorm(deblock1)
        deblock1 = self.decoder_1_6_relu(deblock1)

        #upsampling + concatenation
        deblock1 = F.interpolate(deblock1, scale_factor=2, mode="nearest")
        y2 = torch.cat([deblock1,x5],dim=1) #256 + 128

        #second decoder block
        deblock2 = self.decoder_2_1_conv(y2)

        if (self.adapters):
            adapt20 = self.adapter_20_1_batchnorm(deblock2)
            adapt20 = self.adapter_20_2_conv(adapt20)
            deblock2 = deblock2 + adapt20

        deblock2 = self.decoder_2_2_batchnorm(deblock2)
        deblock2 = self.decoder_2_3_relu(deblock2)
        deblock2 = self.decoder_2_4_conv(deblock2)

        if (self.adapters):
            adapt21 = self.adapter_21_1_batchnorm(deblock2)
            adapt21 = self.adapter_21_2_conv(adapt21)
            deblock2 = deblock2 + adapt21

        deblock2 = self.decoder_2_5_batchnorm(deblock2)
        deblock2 = self.decoder_2_6_relu(deblock2)

        #upsampling + concatenation
        deblock2 = F.interpolate(deblock2, scale_factor=2, mode="nearest")
        y3 = torch.cat([deblock2,x3],dim=1) #128 + 64

        #third decoder block
        deblock3 = self.decoder_3_1_conv(y3)

        if (self.adapters):
            adapt22 = self.adapter_22_1_batchnorm(deblock3)
            adapt22 = self.adapter_22_2_conv(adapt22)
            deblock3 = deblock3 + adapt22

        deblock3 = self.decoder_3_2_batchnorm(deblock3)
        deblock3 = self.decoder_3_3_relu(deblock3)
        deblock3 = self.decoder_3_4_conv(deblock3)

        if (self.adapters):
            adapt23 = self.adapter_23_1_batchnorm(deblock3)
            adapt23 = self.adapter_23_2_conv(adapt23)
            deblock3 = deblock3 + adapt23

        deblock3 = self.decoder_3_5_batchnorm(deblock3)
        deblock3 = self.decoder_3_6_relu(deblock3)

        #upsampling + concatenation
        deblock3 = F.interpolate(deblock3, scale_factor=2, mode="nearest")
        y4 = torch.cat([deblock3,x1],dim=1) #64 + 64

        #fourth decoder block
        deblock4 = self.decoder_4_1_conv(y4)

        if (self.adapters):
            adapt24 = self.adapter_24_1_batchnorm(deblock4)
            adapt24 = self.adapter_24_2_conv(adapt24)
            deblock4 = deblock4 + adapt24

        deblock4 = self.decoder_4_2_batchnorm(deblock4)
        deblock4 = self.decoder_4_3_relu(deblock4)
        deblock4 = self.decoder_4_4_conv(deblock4)

        if (self.adapters):
            adapt25 = self.adapter_25_1_batchnorm(deblock4)
            adapt25 = self.adapter_25_2_conv(adapt25)
            deblock4 = deblock4 + adapt25

        deblock4 = self.decoder_4_5_batchnorm(deblock4)
        deblock4 = self.decoder_4_6_relu(deblock4)

        #no concatenation
        deblock4 = F.interpolate(deblock4, scale_factor=2, mode="nearest")
        y5 = deblock4 #32

        #fifth decoder block (segmentation head)
        deblock5 = self.decoder_5_1_conv(y5)

        if (self.adapters):
            adapt26 = self.adapter_26_1_batchnorm(deblock5)
            adapt26 = self.adapter_26_2_conv(adapt26)
            deblock5 = deblock5 + adapt26

        deblock5 = self.decoder_5_2_batchnorm(deblock5)
        deblock5 = self.decoder_5_3_relu(deblock5)
        deblock5 = self.decoder_5_4_conv(deblock5)

        if (self.adapters):
            adapt27 = self.adapter_27_1_batchnorm(deblock5)
            adapt27 = self.adapter_27_2_conv(adapt27)
            deblock5 = deblock5 + adapt27

        deblock5 = self.decoder_5_5_batchnorm(deblock5)
        deblock5 = self.decoder_5_6_relu(deblock5)

        #no concatenation (no upsampling in seg head)
        #deblock5 = F.interpolate(deblock5, scale_factor=2, mode="nearest")
        y6 = deblock5 #16

        #sixth decoder block
        deblock6 = self.decoder_6_1_conv(y6)

        # print("Decoder Shapes:")
        # print(y1.size(),y2.size(),y3.size(),y4.size(),y5.size(),y6.size(),deblock6.size())

        return deblock6

#resnet18
class Unet_18_relu(nn.Module):

    def __init__(self, in_channels, num_classes, adapters):

        super(Unet_18_relu, self).__init__()
        self.adapters = adapters
        self.adapter_relu = nn.ReLU(inplace=True)

        #----------------------------------------------------------------        
        #encoder layers (encoder_block_layernumber_typeoflayer)
        #----------------------------------------------------------------

        #first block
        self.encoder_1_1_conv = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)   
        self.encoder_1_2_batchnorm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
        self.encoder_1_3_relu = nn.ReLU(inplace=True)
        self.encoder_1_4_maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        #adapter
        if (self.adapters):
            self.adapter_1_1_batchnorm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_1_2_conv = nn.Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_1_2_conv)
            self.init_bnorm(self.adapter_1_1_batchnorm)

        #second block
        self.encoder_2_1_conv = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        #adapter
        if (self.adapters):
            self.adapter_2_1_batchnorm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_2_2_conv = nn.Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_2_2_conv)
            self.init_bnorm(self.adapter_2_1_batchnorm)

        self.encoder_2_2_batchnorm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.encoder_2_3_relu = nn.ReLU(inplace=True)
        self.encoder_2_4_conv = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        #adapter
        if (self.adapters):
            self.adapter_3_1_batchnorm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_3_2_conv = nn.Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_3_2_conv)
            self.init_bnorm(self.adapter_3_1_batchnorm)

        self.encoder_2_5_batchnorm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.encoder_2_6_relu = nn.ReLU(inplace=True)
        self.encoder_2_7_conv = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        #adapter
        if (self.adapters):
            self.adapter_4_1_batchnorm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_4_2_conv = nn.Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_4_2_conv)
            self.init_bnorm(self.adapter_4_1_batchnorm)

        self.encoder_2_8_batchnorm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.encoder_2_9_relu = nn.ReLU(inplace=True)
        self.encoder_2_10_conv = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        #adapter
        if (self.adapters):
            self.adapter_5_1_batchnorm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_5_2_conv = nn.Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_5_2_conv)
            self.init_bnorm(self.adapter_5_1_batchnorm)

        self.encoder_2_11_batchnorm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.encoder_2_12_relu = nn.ReLU(inplace=True)

        #third block
        self.encoder_3_1_conv = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

        #adapter
        if (self.adapters):
            self.adapter_6_1_batchnorm = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_6_2_conv = nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_6_2_conv)
            self.init_bnorm(self.adapter_6_1_batchnorm)

        self.encoder_3_2_batchnorm = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.encoder_3_3_relu = nn.ReLU(inplace=True)
        self.encoder_3_4_conv = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        #adapter
        if (self.adapters):
            self.adapter_7_1_batchnorm = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_7_2_conv = nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_7_2_conv)
            self.init_bnorm(self.adapter_7_1_batchnorm)

        self.encoder_3_5_batchnorm = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        #need it for concatenation
        self.encoder_3_6_downsample = nn.Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
        self.encoder_3_7_batchnorm = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.encoder_3_8_relu = nn.ReLU(inplace=True)

        self.encoder_3_9_conv = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        #adapter
        if (self.adapters):
            self.adapter_8_1_batchnorm = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_8_2_conv = nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_8_2_conv)
            self.init_bnorm(self.adapter_8_1_batchnorm)

        self.encoder_3_10_batchnorm = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.encoder_3_11_relu = nn.ReLU(inplace=True)
        self.encoder_3_12_conv = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        #adapter
        if (self.adapters):
            self.adapter_9_1_batchnorm = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_9_2_conv = nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_9_2_conv)
            self.init_bnorm(self.adapter_9_1_batchnorm)

        self.encoder_3_13_batchnorm = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.encoder_3_14_relu = nn.ReLU(inplace=True)

        #fourth block
        self.encoder_4_1_conv = nn.Conv2d(128,256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

        #adapter
        if (self.adapters):
            self.adapter_10_1_batchnorm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_10_2_conv = nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_10_2_conv)
            self.init_bnorm(self.adapter_10_1_batchnorm)

        self.encoder_4_2_batchnorm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.encoder_4_3_relu = nn.ReLU(inplace=True)
        self.encoder_4_4_conv = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        #adapter
        if (self.adapters):
            self.adapter_11_1_batchnorm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_11_2_conv = nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_11_2_conv)
            self.init_bnorm(self.adapter_11_1_batchnorm)

        self.encoder_4_5_batchnorm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        #need it for concatenation
        self.encoder_4_6_downsample = nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
        self.encoder_4_7_batchnorm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.encoder_4_8_relu = nn.ReLU(inplace=True)

        self.encoder_4_9_conv = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        #adapter
        if (self.adapters):
            self.adapter_12_1_batchnorm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_12_2_conv = nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_12_2_conv)
            self.init_bnorm(self.adapter_12_1_batchnorm)

        self.encoder_4_10_batchnorm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.encoder_4_11_relu = nn.ReLU(inplace=True)
        self.encoder_4_12_conv = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        #adapter
        if (self.adapters):
            self.adapter_13_1_batchnorm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_13_2_conv = nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_13_2_conv)
            self.init_bnorm(self.adapter_13_1_batchnorm)

        self.encoder_4_13_batchnorm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.encoder_4_14_relu = nn.ReLU(inplace=True)

        #fifth block
        self.encoder_5_1_conv = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

        #adapter
        if (self.adapters):
            self.adapter_14_1_batchnorm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_14_2_conv = nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_14_2_conv)
            self.init_bnorm(self.adapter_14_1_batchnorm)

        self.encoder_5_2_batchnorm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.encoder_5_3_relu = nn.ReLU(inplace=True)
        self.encoder_5_4_conv = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        #adapter
        if (self.adapters):
            self.adapter_15_1_batchnorm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_15_2_conv = nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_15_2_conv)
            self.init_bnorm(self.adapter_15_1_batchnorm)

        self.encoder_5_5_batchnorm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        #need it for concatenation
        self.encoder_5_6_downsample = nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
        self.encoder_5_7_batchnorm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.encoder_5_8_relu = nn.ReLU(inplace=True)

        self.encoder_5_9_conv = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        #adapter
        if (self.adapters):
            self.adapter_16_1_batchnorm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_16_2_conv = nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_16_2_conv)
            self.init_bnorm(self.adapter_16_1_batchnorm)

        self.encoder_5_10_batchnorm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.encoder_5_11_relu = nn.ReLU(inplace=True)
        self.encoder_5_12_conv = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        #adapter
        if (self.adapters):
            self.adapter_17_1_batchnorm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_17_2_conv = nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_17_2_conv)
            self.init_bnorm(self.adapter_17_1_batchnorm)

        self.encoder_5_13_batchnorm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.encoder_5_14_relu = nn.ReLU(inplace=True)

        #----------------------------------------------------------------        
        #decoder layers (decoder_block_layernumber_typeoflayer)
        #----------------------------------------------------------------

        #first decoder block
        self.decoder_1_1_conv = nn.Conv2d(768, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        #adapter
        if (self.adapters):
            self.adapter_18_1_batchnorm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_18_2_conv = nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_18_2_conv)
            self.init_bnorm(self.adapter_18_1_batchnorm)

        self.decoder_1_2_batchnorm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.decoder_1_3_relu = nn.ReLU(inplace=True)
        self.decoder_1_4_conv = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        
        #adapter
        if (self.adapters):
            self.adapter_19_1_batchnorm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_19_2_conv = nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_19_2_conv)
            self.init_bnorm(self.adapter_19_1_batchnorm)
        
        self.decoder_1_5_batchnorm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.decoder_1_6_relu = nn.ReLU(inplace=True)

        #second decoder block
        self.decoder_2_1_conv = nn.Conv2d(384, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        #adapter
        if (self.adapters):
            self.adapter_20_1_batchnorm = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_20_2_conv = nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_20_2_conv)
            self.init_bnorm(self.adapter_20_1_batchnorm)

        self.decoder_2_2_batchnorm = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.decoder_2_3_relu = nn.ReLU(inplace=True)
        self.decoder_2_4_conv = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        #adapter
        if (self.adapters):
            self.adapter_21_1_batchnorm = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_21_2_conv = nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_21_2_conv)
            self.init_bnorm(self.adapter_21_1_batchnorm)

        self.decoder_2_5_batchnorm = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.decoder_2_6_relu = nn.ReLU(inplace=True)

        #third decoder block
        self.decoder_3_1_conv = nn.Conv2d(192, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        #adapter
        if (self.adapters):
            self.adapter_22_1_batchnorm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_22_2_conv = nn.Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_22_2_conv)
            self.init_bnorm(self.adapter_22_1_batchnorm)

        self.decoder_3_2_batchnorm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.decoder_3_3_relu = nn.ReLU(inplace=True)
        self.decoder_3_4_conv = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        #adapter
        if (self.adapters):
            self.adapter_23_1_batchnorm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_23_2_conv = nn.Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_23_2_conv)
            self.init_bnorm(self.adapter_23_1_batchnorm)

        self.decoder_3_5_batchnorm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.decoder_3_6_relu = nn.ReLU(inplace=True)

        #fourth decoder block
        self.decoder_4_1_conv = nn.Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        #adapter
        if (self.adapters):
            self.adapter_24_1_batchnorm = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_24_2_conv = nn.Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_24_2_conv)
            self.init_bnorm(self.adapter_24_1_batchnorm)

        self.decoder_4_2_batchnorm = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.decoder_4_3_relu = nn.ReLU(inplace=True)
        self.decoder_4_4_conv = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        #adapter
        if (self.adapters):
            self.adapter_25_1_batchnorm = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_25_2_conv = nn.Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_25_2_conv)
            self.init_bnorm(self.adapter_25_1_batchnorm)

        self.decoder_4_5_batchnorm = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.decoder_4_6_relu = nn.ReLU(inplace=True)

        #fifth decoder block
        self.decoder_5_1_conv = nn.Conv2d(32, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        #adapter
        if (self.adapters):
            self.adapter_26_1_batchnorm = nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_26_2_conv = nn.Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_26_2_conv)
            self.init_bnorm(self.adapter_26_1_batchnorm)

        self.decoder_5_2_batchnorm = nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.decoder_5_3_relu = nn.ReLU(inplace=True)
        self.decoder_5_4_conv = nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        #adapter
        if (self.adapters):
            self.adapter_27_1_batchnorm = nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_27_2_conv = nn.Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_27_2_conv)
            self.init_bnorm(self.adapter_27_1_batchnorm)

        self.decoder_5_5_batchnorm = nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.decoder_5_6_relu = nn.ReLU(inplace=True)

        #final segmentation head
        self.decoder_6_1_conv = nn.Conv2d(16, num_classes, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    def init_convweights(self, layer):
        layer.weight.data.zero_()
        return

    def init_bnorm(self, layer):
        layer.weight.data.fill_(1)
        layer.bias.data.zero_()
        layer.running_mean.zero_()
        layer.running_var.fill_(1)
        layer.num_batches_tracked.zero_()
        return 
    
    def forward(self, x):

        # print("Input Shape:")
        # print(x.size())

        #first block
        block1 = self.encoder_1_1_conv(x)

        if (self.adapters):
            adapt1 = self.adapter_1_1_batchnorm(block1)
            adapt1 = self.adapter_relu(adapt1)
            adapt1 = self.adapter_1_2_conv(adapt1)
            block1 = block1 + adapt1

        block1 = self.encoder_1_2_batchnorm(block1)
        block1 = self.encoder_1_3_relu(block1)
        x1 = block1 #save it for decoder
        block1 = self.encoder_1_4_maxpool(block1)

        #second block
        block2 = self.encoder_2_1_conv(block1)

        if (self.adapters):
            adapt2 = self.adapter_2_1_batchnorm(block2)
            adapt2 = self.adapter_relu(adapt2)
            adapt2 = self.adapter_2_2_conv(adapt2)
            block2 = block2 + adapt2

        block2 = self.encoder_2_2_batchnorm(block2) 
        block2 = self.encoder_2_3_relu(block2)
        block2 = self.encoder_2_4_conv(block2)

        if (self.adapters):
            adapt3 = self.adapter_3_1_batchnorm(block2)
            adapt3 = self.adapter_relu(adapt3)
            adapt3 = self.adapter_3_2_conv(adapt3)
            block2 = block2 + adapt3

        block2 = self.encoder_2_5_batchnorm(block2)

        #skip connection
        x2 = block1 + block2
        x2 = self.encoder_2_6_relu(x2)

        block2 = self.encoder_2_7_conv(x2)

        if (self.adapters):
            adapt4 = self.adapter_4_1_batchnorm(block2)
            adapt4 = self.adapter_relu(adapt4)
            adapt4 = self.adapter_4_2_conv(adapt4)
            block2 = block2 + adapt4

        block2 = self.encoder_2_8_batchnorm(block2) 
        block2 = self.encoder_2_9_relu(block2)
        block2 = self.encoder_2_10_conv(block2)

        if (self.adapters):
            adapt5 = self.adapter_5_1_batchnorm(block2)
            adapt5 = self.adapter_relu(adapt5)
            adapt5 = self.adapter_5_2_conv(adapt5)
            block2 = block2 + adapt5

        block2 = self.encoder_2_11_batchnorm(block2)

        #skip connection
        x3 = x2 + block2
        x3 = self.encoder_2_12_relu(x3)

        #third block
        block3 = self.encoder_3_1_conv(x3)

        if (self.adapters):
            adapt6 = self.adapter_6_1_batchnorm(block3)
            adapt6 = self.adapter_relu(adapt6)
            adapt6 = self.adapter_6_2_conv(adapt6)
            block3 = block3 + adapt6

        block3 = self.encoder_3_2_batchnorm(block3) 
        block3 = self.encoder_3_3_relu(block3) 
        block3 = self.encoder_3_4_conv(block3) 

        if (self.adapters):
            adapt7 = self.adapter_7_1_batchnorm(block3)
            adapt7 = self.adapter_relu(adapt7)
            adapt7 = self.adapter_7_2_conv(adapt7)
            block3 = block3 + adapt7

        block3 = self.encoder_3_5_batchnorm(block3) 

        #need it for concatenation
        x4 = self.encoder_3_6_downsample(x3) 
        x4 = self.encoder_3_7_batchnorm(x4)
        x4 = x4 + block3
        x4 = self.encoder_3_8_relu(x4) 

        block3 = self.encoder_3_9_conv(x4) 

        if (self.adapters):
            adapt8 = self.adapter_8_1_batchnorm(block3)
            adapt8 = self.adapter_relu(adapt8)
            adapt8 = self.adapter_8_2_conv(adapt8)
            block3 = block3 + adapt8

        block3 = self.encoder_3_10_batchnorm(block3)
        block3 = self.encoder_3_11_relu(block3)
        block3 = self.encoder_3_12_conv(block3)

        if (self.adapters):
            adapt9 = self.adapter_9_1_batchnorm(block3)
            adapt9 = self.adapter_relu(adapt9)
            adapt9 = self.adapter_9_2_conv(adapt9)
            block3 = block3 + adapt9

        block3 = self.encoder_3_13_batchnorm(block3)

        #skip connection
        x5 = x4 + block3
        x5 = self.encoder_3_14_relu(x5)

        #fourth block
        block4 = self.encoder_4_1_conv(x5)

        if (self.adapters):
            adapt10 = self.adapter_10_1_batchnorm(block4)
            adapt10 = self.adapter_relu(adapt10)
            adapt10 = self.adapter_10_2_conv(adapt10)
            block4 = block4 + adapt10

        block4 = self.encoder_4_2_batchnorm(block4) 
        block4 = self.encoder_4_3_relu(block4) 
        block4 = self.encoder_4_4_conv(block4) 

        if (self.adapters):
            adapt11 = self.adapter_11_1_batchnorm(block4)
            adapt11 = self.adapter_relu(adapt11)
            adapt11 = self.adapter_11_2_conv(adapt11)
            block4 = block4 + adapt11

        block4 = self.encoder_4_5_batchnorm(block4) 

        #need it for concatenation
        x6 = self.encoder_4_6_downsample(x5) 
        x6 = self.encoder_4_7_batchnorm(x6)
        x6 = x6 + block4
        x6 = self.encoder_4_8_relu(x6)

        block4 = self.encoder_4_9_conv(x6) 

        if (self.adapters):
            adapt12 = self.adapter_12_1_batchnorm(block4)
            adapt12 = self.adapter_relu(adapt12)
            adapt12 = self.adapter_12_2_conv(adapt12)
            block4 = block4 + adapt12

        block4 = self.encoder_4_10_batchnorm(block4)
        block4 = self.encoder_4_11_relu(block4)
        block4 = self.encoder_4_12_conv(block4)

        if (self.adapters):
            adapt13 = self.adapter_13_1_batchnorm(block4)
            adapt13 = self.adapter_relu(adapt13)
            adapt13 = self.adapter_13_2_conv(adapt13)
            block4 = block4 + adapt13

        block4 = self.encoder_4_13_batchnorm(block4)
        
        #skip connection
        x7 = x6 + block4
        x7 = self.encoder_4_14_relu(x7)

        #fifth block
        block5 = self.encoder_5_1_conv(x7)

        if (self.adapters):
            adapt14 = self.adapter_14_1_batchnorm(block5)
            adapt14 = self.adapter_relu(adapt14)
            adapt14 = self.adapter_14_2_conv(adapt14)
            block5 = block5 + adapt14

        block5 = self.encoder_5_2_batchnorm(block5) 
        block5 = self.encoder_5_3_relu(block5) 
        block5 = self.encoder_5_4_conv(block5) 

        if (self.adapters):
            adapt15 = self.adapter_15_1_batchnorm(block5)
            adapt15 = self.adapter_relu(adapt15)
            adapt15 = self.adapter_15_2_conv(adapt15)
            block5 = block5 + adapt15

        block5 = self.encoder_5_5_batchnorm(block5) 

        #need it for concatenation
        x8 = self.encoder_5_6_downsample(x7) 
        x8 = self.encoder_5_7_batchnorm(x8)
        x8 = x8 + block5
        x8 = self.encoder_5_8_relu(x8)

        block5 = self.encoder_5_9_conv(x8) 

        if (self.adapters):
            adapt16 = self.adapter_16_1_batchnorm(block5)
            adapt16 = self.adapter_relu(adapt16)
            adapt16 = self.adapter_16_2_conv(adapt16)
            block5 = block5 + adapt16

        block5 = self.encoder_5_10_batchnorm(block5)
        block5 = self.encoder_5_11_relu(block5)
        block5 = self.encoder_5_12_conv(block5)

        if (self.adapters):
            adapt17 = self.adapter_17_1_batchnorm(block5)
            adapt17 = self.adapter_relu(adapt17)
            adapt17 = self.adapter_17_2_conv(adapt17)
            block5 = block5 + adapt17

        block5 = self.encoder_5_13_batchnorm(block5)

        #skip connection
        x9 = x8 + block5 #bottleneck
        x9 = self.encoder_5_14_relu(x9)

        # print("Encoder Shapes: ")
        # print(x1.size(),block2.size(),block3.size(),block4.size(),block5.size())

        #upsampling + concatenation
        y1 = F.interpolate(x9, scale_factor=2, mode="nearest")
        y1 = torch.cat([y1,x7],dim=1) #512 + 256

        #first decoder block
        deblock1 = self.decoder_1_1_conv(y1)

        if (self.adapters):
            adapt18 = self.adapter_18_1_batchnorm(deblock1)
            adapt18 = self.adapter_relu(adapt18)
            adapt18 = self.adapter_18_2_conv(adapt18)
            deblock1 = deblock1 + adapt18

        deblock1 = self.decoder_1_2_batchnorm(deblock1)
        deblock1 = self.decoder_1_3_relu(deblock1)
        deblock1 = self.decoder_1_4_conv(deblock1)

        if (self.adapters):
            adapt19 = self.adapter_19_1_batchnorm(deblock1)
            adapt19 = self.adapter_relu(adapt19)
            adapt19 = self.adapter_19_2_conv(adapt19)
            deblock1 = deblock1 + adapt19

        deblock1 = self.decoder_1_5_batchnorm(deblock1)
        deblock1 = self.decoder_1_6_relu(deblock1)

        #upsampling + concatenation
        deblock1 = F.interpolate(deblock1, scale_factor=2, mode="nearest")
        y2 = torch.cat([deblock1,x5],dim=1) #256 + 128

        #second decoder block
        deblock2 = self.decoder_2_1_conv(y2)

        if (self.adapters):
            adapt20 = self.adapter_20_1_batchnorm(deblock2)
            adapt20 = self.adapter_relu(adapt20)
            adapt20 = self.adapter_20_2_conv(adapt20)
            deblock2 = deblock2 + adapt20

        deblock2 = self.decoder_2_2_batchnorm(deblock2)
        deblock2 = self.decoder_2_3_relu(deblock2)
        deblock2 = self.decoder_2_4_conv(deblock2)

        if (self.adapters):
            adapt21 = self.adapter_21_1_batchnorm(deblock2)
            adapt21 = self.adapter_relu(adapt21)
            adapt21 = self.adapter_21_2_conv(adapt21)
            deblock2 = deblock2 + adapt21

        deblock2 = self.decoder_2_5_batchnorm(deblock2)
        deblock2 = self.decoder_2_6_relu(deblock2)

        #upsampling + concatenation
        deblock2 = F.interpolate(deblock2, scale_factor=2, mode="nearest")
        y3 = torch.cat([deblock2,x3],dim=1) #128 + 64

        #third decoder block
        deblock3 = self.decoder_3_1_conv(y3)

        if (self.adapters):
            adapt22 = self.adapter_22_1_batchnorm(deblock3)
            adapt22 = self.adapter_relu(adapt22)
            adapt22 = self.adapter_22_2_conv(adapt22)
            deblock3 = deblock3 + adapt22

        deblock3 = self.decoder_3_2_batchnorm(deblock3)
        deblock3 = self.decoder_3_3_relu(deblock3)
        deblock3 = self.decoder_3_4_conv(deblock3)

        if (self.adapters):
            adapt23 = self.adapter_23_1_batchnorm(deblock3)
            adapt23 = self.adapter_relu(adapt23)
            adapt23 = self.adapter_23_2_conv(adapt23)
            deblock3 = deblock3 + adapt23

        deblock3 = self.decoder_3_5_batchnorm(deblock3)
        deblock3 = self.decoder_3_6_relu(deblock3)

        #upsampling + concatenation
        deblock3 = F.interpolate(deblock3, scale_factor=2, mode="nearest")
        y4 = torch.cat([deblock3,x1],dim=1) #64 + 64

        #fourth decoder block
        deblock4 = self.decoder_4_1_conv(y4)

        if (self.adapters):
            adapt24 = self.adapter_24_1_batchnorm(deblock4)
            adapt24 = self.adapter_relu(adapt24)
            adapt24 = self.adapter_24_2_conv(adapt24)
            deblock4 = deblock4 + adapt24

        deblock4 = self.decoder_4_2_batchnorm(deblock4)
        deblock4 = self.decoder_4_3_relu(deblock4)
        deblock4 = self.decoder_4_4_conv(deblock4)

        if (self.adapters):
            adapt25 = self.adapter_25_1_batchnorm(deblock4)
            adapt25 = self.adapter_relu(adapt25)
            adapt25 = self.adapter_25_2_conv(adapt25)
            deblock4 = deblock4 + adapt25

        deblock4 = self.decoder_4_5_batchnorm(deblock4)
        deblock4 = self.decoder_4_6_relu(deblock4)

        #no concatenation
        deblock4 = F.interpolate(deblock4, scale_factor=2, mode="nearest")
        y5 = deblock4 #32

        #fifth decoder block (segmentation head)
        deblock5 = self.decoder_5_1_conv(y5)

        if (self.adapters):
            adapt26 = self.adapter_26_1_batchnorm(deblock5)
            adapt26 = self.adapter_relu(adapt26)
            adapt26 = self.adapter_26_2_conv(adapt26)
            deblock5 = deblock5 + adapt26

        deblock5 = self.decoder_5_2_batchnorm(deblock5)
        deblock5 = self.decoder_5_3_relu(deblock5)
        deblock5 = self.decoder_5_4_conv(deblock5)

        if (self.adapters):
            adapt27 = self.adapter_27_1_batchnorm(deblock5)
            adapt27 = self.adapter_relu(adapt27)
            adapt27 = self.adapter_27_2_conv(adapt27)
            deblock5 = deblock5 + adapt27

        deblock5 = self.decoder_5_5_batchnorm(deblock5)
        deblock5 = self.decoder_5_6_relu(deblock5)

        #no concatenation (no upsampling in seg head)
        #deblock5 = F.interpolate(deblock5, scale_factor=2, mode="nearest")
        y6 = deblock5 #16

        #sixth decoder block
        deblock6 = self.decoder_6_1_conv(y6)

        # print("Decoder Shapes:")
        # print(y1.size(),y2.size(),y3.size(),y4.size(),y5.size(),y6.size(),deblock6.size())

        return deblock6

#resnet18
class Unet_18_conv(nn.Module):

    def __init__(self, in_channels, num_classes, adapters):

        super(Unet_18_conv, self).__init__()
        self.adapters = adapters
        self.adapter_relu = nn.ReLU(inplace=True)

        #----------------------------------------------------------------        
        #encoder layers (encoder_block_layernumber_typeoflayer)
        #----------------------------------------------------------------

        #first block
        self.encoder_1_1_conv = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)   
        self.encoder_1_2_batchnorm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
        self.encoder_1_3_relu = nn.ReLU(inplace=True)
        self.encoder_1_4_maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        #adapter
        if (self.adapters):
            self.adapter_1_1_batchnorm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_1_2_conv = nn.Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_1_2_conv)
            self.init_bnorm(self.adapter_1_1_batchnorm)

        #second block
        self.encoder_2_1_conv = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        #adapter
        if (self.adapters):
            self.adapter_2_1_batchnorm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_2_2_conv = nn.Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_2_2_conv)
            self.init_bnorm(self.adapter_2_1_batchnorm)

        self.encoder_2_2_batchnorm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.encoder_2_3_relu = nn.ReLU(inplace=True)
        self.encoder_2_4_conv = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        #adapter
        if (self.adapters):
            self.adapter_3_1_batchnorm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_3_2_conv = nn.Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_3_2_conv)
            self.init_bnorm(self.adapter_3_1_batchnorm)

        self.encoder_2_5_batchnorm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.encoder_2_6_relu = nn.ReLU(inplace=True)
        self.encoder_2_7_conv = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        #adapter
        if (self.adapters):
            self.adapter_4_1_batchnorm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_4_2_conv = nn.Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_4_2_conv)
            self.init_bnorm(self.adapter_4_1_batchnorm)

        self.encoder_2_8_batchnorm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.encoder_2_9_relu = nn.ReLU(inplace=True)
        self.encoder_2_10_conv = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        #adapter
        if (self.adapters):
            self.adapter_5_1_batchnorm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_5_2_conv = nn.Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_5_2_conv)
            self.init_bnorm(self.adapter_5_1_batchnorm)

        self.encoder_2_11_batchnorm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.encoder_2_12_relu = nn.ReLU(inplace=True)

        #third block
        self.encoder_3_1_conv = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

        #adapter
        if (self.adapters):
            self.adapter_6_1_batchnorm = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_6_2_conv = nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_6_2_conv)
            self.init_bnorm(self.adapter_6_1_batchnorm)

        self.encoder_3_2_batchnorm = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.encoder_3_3_relu = nn.ReLU(inplace=True)
        self.encoder_3_4_conv = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        #adapter
        if (self.adapters):
            self.adapter_7_1_batchnorm = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_7_2_conv = nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_7_2_conv)
            self.init_bnorm(self.adapter_7_1_batchnorm)

        self.encoder_3_5_batchnorm = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        #need it for concatenation
        self.encoder_3_6_downsample = nn.Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
        self.encoder_3_7_batchnorm = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.encoder_3_8_relu = nn.ReLU(inplace=True)

        self.encoder_3_9_conv = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        #adapter
        if (self.adapters):
            self.adapter_8_1_batchnorm = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_8_2_conv = nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_8_2_conv)
            self.init_bnorm(self.adapter_8_1_batchnorm)

        self.encoder_3_10_batchnorm = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.encoder_3_11_relu = nn.ReLU(inplace=True)
        self.encoder_3_12_conv = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        #adapter
        if (self.adapters):
            self.adapter_9_1_batchnorm = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_9_2_conv = nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_9_2_conv)
            self.init_bnorm(self.adapter_9_1_batchnorm)

        self.encoder_3_13_batchnorm = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.encoder_3_14_relu = nn.ReLU(inplace=True)

        #fourth block
        self.encoder_4_1_conv = nn.Conv2d(128,256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

        #adapter
        if (self.adapters):
            self.adapter_10_1_batchnorm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_10_2_conv = nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_10_2_conv)
            self.init_bnorm(self.adapter_10_1_batchnorm)

        self.encoder_4_2_batchnorm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.encoder_4_3_relu = nn.ReLU(inplace=True)
        self.encoder_4_4_conv = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        #adapter
        if (self.adapters):
            self.adapter_11_1_batchnorm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_11_2_conv = nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_11_2_conv)
            self.init_bnorm(self.adapter_11_1_batchnorm)

        self.encoder_4_5_batchnorm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        #need it for concatenation
        self.encoder_4_6_downsample = nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
        self.encoder_4_7_batchnorm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.encoder_4_8_relu = nn.ReLU(inplace=True)

        self.encoder_4_9_conv = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        #adapter
        if (self.adapters):
            self.adapter_12_1_batchnorm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_12_2_conv = nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_12_2_conv)
            self.init_bnorm(self.adapter_12_1_batchnorm)

        self.encoder_4_10_batchnorm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.encoder_4_11_relu = nn.ReLU(inplace=True)
        self.encoder_4_12_conv = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        #adapter
        if (self.adapters):
            self.adapter_13_1_batchnorm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_13_2_conv = nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_13_2_conv)
            self.init_bnorm(self.adapter_13_1_batchnorm)

        self.encoder_4_13_batchnorm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.encoder_4_14_relu = nn.ReLU(inplace=True)

        #fifth block
        self.encoder_5_1_conv = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

        #adapter
        if (self.adapters):
            self.adapter_14_1_batchnorm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_14_2_conv = nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_14_2_conv)
            self.init_bnorm(self.adapter_14_1_batchnorm)

        self.encoder_5_2_batchnorm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.encoder_5_3_relu = nn.ReLU(inplace=True)
        self.encoder_5_4_conv = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        #adapter
        if (self.adapters):
            self.adapter_15_1_batchnorm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_15_2_conv = nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_15_2_conv)
            self.init_bnorm(self.adapter_15_1_batchnorm)

        self.encoder_5_5_batchnorm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        #need it for concatenation
        self.encoder_5_6_downsample = nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
        self.encoder_5_7_batchnorm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.encoder_5_8_relu = nn.ReLU(inplace=True)

        self.encoder_5_9_conv = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        #adapter
        if (self.adapters):
            self.adapter_16_1_batchnorm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_16_2_conv = nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_16_2_conv)
            self.init_bnorm(self.adapter_16_1_batchnorm)

        self.encoder_5_10_batchnorm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.encoder_5_11_relu = nn.ReLU(inplace=True)
        self.encoder_5_12_conv = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        #adapter
        if (self.adapters):
            self.adapter_17_1_batchnorm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_17_2_conv = nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_17_2_conv)
            self.init_bnorm(self.adapter_17_1_batchnorm)

        self.encoder_5_13_batchnorm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.encoder_5_14_relu = nn.ReLU(inplace=True)

        #----------------------------------------------------------------        
        #decoder layers (decoder_block_layernumber_typeoflayer)
        #----------------------------------------------------------------

        #first decoder block
        self.decoder_1_1_conv = nn.Conv2d(768, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        #adapter
        if (self.adapters):
            self.adapter_18_1_batchnorm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_18_2_conv = nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_18_2_conv)
            self.init_bnorm(self.adapter_18_1_batchnorm)

        self.decoder_1_2_batchnorm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.decoder_1_3_relu = nn.ReLU(inplace=True)
        self.decoder_1_4_conv = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        
        #adapter
        if (self.adapters):
            self.adapter_19_1_batchnorm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_19_2_conv = nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_19_2_conv)
            self.init_bnorm(self.adapter_19_1_batchnorm)
        
        self.decoder_1_5_batchnorm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.decoder_1_6_relu = nn.ReLU(inplace=True)

        #second decoder block
        self.decoder_2_1_conv = nn.Conv2d(384, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        #adapter
        if (self.adapters):
            self.adapter_20_1_batchnorm = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_20_2_conv = nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_20_2_conv)
            self.init_bnorm(self.adapter_20_1_batchnorm)

        self.decoder_2_2_batchnorm = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.decoder_2_3_relu = nn.ReLU(inplace=True)
        self.decoder_2_4_conv = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        #adapter
        if (self.adapters):
            self.adapter_21_1_batchnorm = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_21_2_conv = nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_21_2_conv)
            self.init_bnorm(self.adapter_21_1_batchnorm)

        self.decoder_2_5_batchnorm = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.decoder_2_6_relu = nn.ReLU(inplace=True)

        #third decoder block
        self.decoder_3_1_conv = nn.Conv2d(192, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        #adapter
        if (self.adapters):
            self.adapter_22_1_batchnorm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_22_2_conv = nn.Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_22_2_conv)
            self.init_bnorm(self.adapter_22_1_batchnorm)

        self.decoder_3_2_batchnorm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.decoder_3_3_relu = nn.ReLU(inplace=True)
        self.decoder_3_4_conv = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        #adapter
        if (self.adapters):
            self.adapter_23_1_batchnorm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_23_2_conv = nn.Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_23_2_conv)
            self.init_bnorm(self.adapter_23_1_batchnorm)

        self.decoder_3_5_batchnorm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.decoder_3_6_relu = nn.ReLU(inplace=True)

        #fourth decoder block
        self.decoder_4_1_conv = nn.Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        #adapter
        if (self.adapters):
            self.adapter_24_1_batchnorm = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_24_2_conv = nn.Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_24_2_conv)
            self.init_bnorm(self.adapter_24_1_batchnorm)

        self.decoder_4_2_batchnorm = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.decoder_4_3_relu = nn.ReLU(inplace=True)
        self.decoder_4_4_conv = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        #adapter
        if (self.adapters):
            self.adapter_25_1_batchnorm = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_25_2_conv = nn.Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_25_2_conv)
            self.init_bnorm(self.adapter_25_1_batchnorm)

        self.decoder_4_5_batchnorm = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.decoder_4_6_relu = nn.ReLU(inplace=True)

        #fifth decoder block
        self.decoder_5_1_conv = nn.Conv2d(32, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        #adapter
        if (self.adapters):
            self.adapter_26_1_batchnorm = nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_26_2_conv = nn.Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_26_2_conv)
            self.init_bnorm(self.adapter_26_1_batchnorm)

        self.decoder_5_2_batchnorm = nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.decoder_5_3_relu = nn.ReLU(inplace=True)
        self.decoder_5_4_conv = nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        #adapter
        if (self.adapters):
            self.adapter_27_1_batchnorm = nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_27_2_conv = nn.Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_27_2_conv)
            self.init_bnorm(self.adapter_27_1_batchnorm)

        self.decoder_5_5_batchnorm = nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.decoder_5_6_relu = nn.ReLU(inplace=True)

        #final segmentation head
        self.decoder_6_1_conv = nn.Conv2d(16, num_classes, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    def init_convweights(self, layer):
        layer.weight.data.zero_()
        return

    def init_bnorm(self, layer):
        layer.weight.data.fill_(1)
        layer.bias.data.zero_()
        layer.running_mean.zero_()
        layer.running_var.fill_(1)
        layer.num_batches_tracked.zero_()
        return 
    
    def forward(self, x):

        # print("Input Shape:")
        # print(x.size())

        #first block
        block1 = self.encoder_1_1_conv(x)

        if (self.adapters):
            adapt1 = self.adapter_1_2_conv(block1)
            adapt1 = self.adapter_1_1_batchnorm(adapt1)
            block1 = block1 + adapt1

        block1 = self.encoder_1_2_batchnorm(block1)
        block1 = self.encoder_1_3_relu(block1)
        x1 = block1 #save it for decoder
        block1 = self.encoder_1_4_maxpool(block1)

        #second block
        block2 = self.encoder_2_1_conv(block1)

        if (self.adapters):
            adapt2 = self.adapter_2_2_conv(block2)
            adapt2 = self.adapter_2_1_batchnorm(adapt2)
            block2 = block2 + adapt2

        block2 = self.encoder_2_2_batchnorm(block2) 
        block2 = self.encoder_2_3_relu(block2)
        block2 = self.encoder_2_4_conv(block2)

        if (self.adapters):
            adapt3 = self.adapter_3_2_conv(block2)
            adapt3 = self.adapter_3_1_batchnorm(adapt3)
            block2 = block2 + adapt3

        block2 = self.encoder_2_5_batchnorm(block2)

        #skip connection
        x2 = block1 + block2
        x2 = self.encoder_2_6_relu(x2)

        block2 = self.encoder_2_7_conv(x2)

        if (self.adapters):
            adapt4 = self.adapter_4_2_conv(block2)
            adapt4 = self.adapter_4_1_batchnorm(adapt4)
            block2 = block2 + adapt4

        block2 = self.encoder_2_8_batchnorm(block2) 
        block2 = self.encoder_2_9_relu(block2)
        block2 = self.encoder_2_10_conv(block2)

        if (self.adapters):
            adapt5 = self.adapter_5_2_conv(block2)
            adapt5 = self.adapter_5_1_batchnorm(adapt5)
            block2 = block2 + adapt5

        block2 = self.encoder_2_11_batchnorm(block2)

        #skip connection
        x3 = x2 + block2
        x3 = self.encoder_2_12_relu(x3)

        #third block
        block3 = self.encoder_3_1_conv(x3)

        if (self.adapters):
            adapt6 = self.adapter_6_2_conv(block3)
            adapt6 = self.adapter_6_1_batchnorm(adapt6)
            block3 = block3 + adapt6

        block3 = self.encoder_3_2_batchnorm(block3) 
        block3 = self.encoder_3_3_relu(block3) 
        block3 = self.encoder_3_4_conv(block3) 

        if (self.adapters):
            adapt7 = self.adapter_7_2_conv(block3)
            adapt7 = self.adapter_7_1_batchnorm(adapt7)
            block3 = block3 + adapt7

        block3 = self.encoder_3_5_batchnorm(block3) 

        #need it for concatenation
        x4 = self.encoder_3_6_downsample(x3) 
        x4 = self.encoder_3_7_batchnorm(x4)
        x4 = x4 + block3
        x4 = self.encoder_3_8_relu(x4) 

        block3 = self.encoder_3_9_conv(x4) 

        if (self.adapters):
            adapt8 = self.adapter_8_2_conv(block3)
            adapt8 = self.adapter_8_1_batchnorm(adapt8)
            block3 = block3 + adapt8

        block3 = self.encoder_3_10_batchnorm(block3)
        block3 = self.encoder_3_11_relu(block3)
        block3 = self.encoder_3_12_conv(block3)

        if (self.adapters):
            adapt9 = self.adapter_9_2_conv(block3)
            adapt9 = self.adapter_9_1_batchnorm(adapt9)
            block3 = block3 + adapt9

        block3 = self.encoder_3_13_batchnorm(block3)

        #skip connection
        x5 = x4 + block3
        x5 = self.encoder_3_14_relu(x5)

        #fourth block
        block4 = self.encoder_4_1_conv(x5)

        if (self.adapters):
            adapt10 = self.adapter_10_2_conv(block4)
            adapt10 = self.adapter_10_1_batchnorm(adapt10)
            block4 = block4 + adapt10

        block4 = self.encoder_4_2_batchnorm(block4) 
        block4 = self.encoder_4_3_relu(block4) 
        block4 = self.encoder_4_4_conv(block4) 

        if (self.adapters):
            adapt11 = self.adapter_11_2_conv(block4)
            adapt11 = self.adapter_11_1_batchnorm(adapt11)
            block4 = block4 + adapt11

        block4 = self.encoder_4_5_batchnorm(block4) 

        #need it for concatenation
        x6 = self.encoder_4_6_downsample(x5) 
        x6 = self.encoder_4_7_batchnorm(x6)
        x6 = x6 + block4
        x6 = self.encoder_4_8_relu(x6)

        block4 = self.encoder_4_9_conv(x6) 

        if (self.adapters):
            adapt12 = self.adapter_12_2_conv(block4)
            adapt12 = self.adapter_12_1_batchnorm(adapt12)
            block4 = block4 + adapt12

        block4 = self.encoder_4_10_batchnorm(block4)
        block4 = self.encoder_4_11_relu(block4)
        block4 = self.encoder_4_12_conv(block4)

        if (self.adapters):
            adapt13 = self.adapter_13_2_conv(block4)
            adapt13 = self.adapter_13_1_batchnorm(adapt13)
            block4 = block4 + adapt13

        block4 = self.encoder_4_13_batchnorm(block4)
        
        #skip connection
        x7 = x6 + block4
        x7 = self.encoder_4_14_relu(x7)

        #fifth block
        block5 = self.encoder_5_1_conv(x7)

        if (self.adapters):
            adapt14 = self.adapter_14_2_conv(block5)
            adapt14 = self.adapter_14_1_batchnorm(adapt14)
            block5 = block5 + adapt14

        block5 = self.encoder_5_2_batchnorm(block5) 
        block5 = self.encoder_5_3_relu(block5) 
        block5 = self.encoder_5_4_conv(block5) 

        if (self.adapters):
            adapt15 = self.adapter_15_2_conv(block5)
            adapt15 = self.adapter_15_1_batchnorm(adapt15)
            block5 = block5 + adapt15

        block5 = self.encoder_5_5_batchnorm(block5) 

        #need it for concatenation
        x8 = self.encoder_5_6_downsample(x7) 
        x8 = self.encoder_5_7_batchnorm(x8)
        x8 = x8 + block5
        x8 = self.encoder_5_8_relu(x8)

        block5 = self.encoder_5_9_conv(x8) 

        if (self.adapters):
            adapt16 = self.adapter_16_2_conv(block5)
            adapt16 = self.adapter_16_1_batchnorm(adapt16)
            block5 = block5 + adapt16

        block5 = self.encoder_5_10_batchnorm(block5)
        block5 = self.encoder_5_11_relu(block5)
        block5 = self.encoder_5_12_conv(block5)

        if (self.adapters):
            adapt17 = self.adapter_17_2_conv(block5)
            adapt17 = self.adapter_17_1_batchnorm(adapt17)
            block5 = block5 + adapt17

        block5 = self.encoder_5_13_batchnorm(block5)

        #skip connection
        x9 = x8 + block5 #bottleneck
        x9 = self.encoder_5_14_relu(x9)

        # print("Encoder Shapes: ")
        # print(x1.size(),block2.size(),block3.size(),block4.size(),block5.size())

        #upsampling + concatenation
        y1 = F.interpolate(x9, scale_factor=2, mode="nearest")
        y1 = torch.cat([y1,x7],dim=1) #512 + 256

        #first decoder block
        deblock1 = self.decoder_1_1_conv(y1)

        if (self.adapters):
            adapt18 = self.adapter_18_2_conv(deblock1)
            adapt18 = self.adapter_18_1_batchnorm(adapt18)
            deblock1 = deblock1 + adapt18

        deblock1 = self.decoder_1_2_batchnorm(deblock1)
        deblock1 = self.decoder_1_3_relu(deblock1)
        deblock1 = self.decoder_1_4_conv(deblock1)

        if (self.adapters):
            adapt19 = self.adapter_19_2_conv(deblock1)
            adapt19 = self.adapter_19_1_batchnorm(adapt19)
            deblock1 = deblock1 + adapt19

        deblock1 = self.decoder_1_5_batchnorm(deblock1)
        deblock1 = self.decoder_1_6_relu(deblock1)

        #upsampling + concatenation
        deblock1 = F.interpolate(deblock1, scale_factor=2, mode="nearest")
        y2 = torch.cat([deblock1,x5],dim=1) #256 + 128

        #second decoder block
        deblock2 = self.decoder_2_1_conv(y2)

        if (self.adapters):
            adapt20 = self.adapter_20_2_conv(deblock2)
            adapt20 = self.adapter_20_1_batchnorm(adapt20)
            deblock2 = deblock2 + adapt20

        deblock2 = self.decoder_2_2_batchnorm(deblock2)
        deblock2 = self.decoder_2_3_relu(deblock2)
        deblock2 = self.decoder_2_4_conv(deblock2)

        if (self.adapters):
            adapt21 = self.adapter_21_2_conv(deblock2)
            adapt21 = self.adapter_21_1_batchnorm(adapt21)
            deblock2 = deblock2 + adapt21

        deblock2 = self.decoder_2_5_batchnorm(deblock2)
        deblock2 = self.decoder_2_6_relu(deblock2)

        #upsampling + concatenation
        deblock2 = F.interpolate(deblock2, scale_factor=2, mode="nearest")
        y3 = torch.cat([deblock2,x3],dim=1) #128 + 64

        #third decoder block
        deblock3 = self.decoder_3_1_conv(y3)

        if (self.adapters):
            adapt22 = self.adapter_22_2_conv(deblock3)
            adapt22 = self.adapter_22_1_batchnorm(adapt22)
            deblock3 = deblock3 + adapt22

        deblock3 = self.decoder_3_2_batchnorm(deblock3)
        deblock3 = self.decoder_3_3_relu(deblock3)
        deblock3 = self.decoder_3_4_conv(deblock3)

        if (self.adapters):
            adapt23 = self.adapter_23_2_conv(deblock3)
            adapt23 = self.adapter_23_1_batchnorm(adapt23)
            deblock3 = deblock3 + adapt23

        deblock3 = self.decoder_3_5_batchnorm(deblock3)
        deblock3 = self.decoder_3_6_relu(deblock3)

        #upsampling + concatenation
        deblock3 = F.interpolate(deblock3, scale_factor=2, mode="nearest")
        y4 = torch.cat([deblock3,x1],dim=1) #64 + 64

        #fourth decoder block
        deblock4 = self.decoder_4_1_conv(y4)

        if (self.adapters):
            adapt24 = self.adapter_24_2_conv(deblock4)
            adapt24 = self.adapter_24_1_batchnorm(adapt24)
            deblock4 = deblock4 + adapt24

        deblock4 = self.decoder_4_2_batchnorm(deblock4)
        deblock4 = self.decoder_4_3_relu(deblock4)
        deblock4 = self.decoder_4_4_conv(deblock4)

        if (self.adapters):
            adapt25 = self.adapter_25_2_conv(deblock4)
            adapt25 = self.adapter_25_1_batchnorm(adapt25)
            deblock4 = deblock4 + adapt25

        deblock4 = self.decoder_4_5_batchnorm(deblock4)
        deblock4 = self.decoder_4_6_relu(deblock4)

        #no concatenation
        deblock4 = F.interpolate(deblock4, scale_factor=2, mode="nearest")
        y5 = deblock4 #32

        #fifth decoder block (segmentation head)
        deblock5 = self.decoder_5_1_conv(y5)

        if (self.adapters):
            adapt26 = self.adapter_26_2_conv(deblock5)
            adapt26 = self.adapter_26_1_batchnorm(adapt26)
            deblock5 = deblock5 + adapt26

        deblock5 = self.decoder_5_2_batchnorm(deblock5)
        deblock5 = self.decoder_5_3_relu(deblock5)
        deblock5 = self.decoder_5_4_conv(deblock5)

        if (self.adapters):
            adapt27 = self.adapter_27_2_conv(deblock5)
            adapt27 = self.adapter_27_1_batchnorm(adapt27)
            deblock5 = deblock5 + adapt27

        deblock5 = self.decoder_5_5_batchnorm(deblock5)
        deblock5 = self.decoder_5_6_relu(deblock5)

        #no concatenation (no upsampling in seg head)
        #deblock5 = F.interpolate(deblock5, scale_factor=2, mode="nearest")
        y6 = deblock5 #16

        #sixth decoder block
        deblock6 = self.decoder_6_1_conv(y6)

        # print("Decoder Shapes:")
        # print(y1.size(),y2.size(),y3.size(),y4.size(),y5.size(),y6.size(),deblock6.size())

        return deblock6

#resnet18
class Unet_18_rep2(nn.Module):

    def __init__(self, in_channels, num_classes, adapters):

        super(Unet_18_rep2, self).__init__()
        self.adapters = adapters

        #----------------------------------------------------------------        
        #encoder layers (encoder_block_layernumber_typeoflayer)
        #----------------------------------------------------------------

        #first block
        self.encoder_1_1_conv = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)   
        self.encoder_1_2_batchnorm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
        self.encoder_1_3_relu = nn.ReLU(inplace=True)
        self.encoder_1_4_maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        #adapter
        if (self.adapters):
            self.adapter_1_1_batchnorm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_1_2_conv = nn.Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_1_2_conv)
            self.init_bnorm(self.adapter_1_1_batchnorm)
            self.adapter_1_3_batchnorm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_1_4_conv = nn.Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_1_4_conv)
            self.init_bnorm(self.adapter_1_3_batchnorm)

        #second block
        self.encoder_2_1_conv = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        #adapter
        if (self.adapters):
            self.adapter_2_1_batchnorm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_2_2_conv = nn.Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_2_2_conv)
            self.init_bnorm(self.adapter_2_1_batchnorm)
            self.adapter_2_3_batchnorm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_2_4_conv = nn.Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_2_4_conv)
            self.init_bnorm(self.adapter_2_3_batchnorm)

        self.encoder_2_2_batchnorm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.encoder_2_3_relu = nn.ReLU(inplace=True)
        self.encoder_2_4_conv = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        #adapter
        if (self.adapters):
            self.adapter_3_1_batchnorm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_3_2_conv = nn.Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_3_2_conv)
            self.init_bnorm(self.adapter_3_1_batchnorm)
            self.adapter_3_3_batchnorm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_3_4_conv = nn.Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_3_4_conv)
            self.init_bnorm(self.adapter_3_3_batchnorm)

        self.encoder_2_5_batchnorm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.encoder_2_6_relu = nn.ReLU(inplace=True)
        self.encoder_2_7_conv = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        #adapter
        if (self.adapters):
            self.adapter_4_1_batchnorm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_4_2_conv = nn.Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_4_2_conv)
            self.init_bnorm(self.adapter_4_1_batchnorm)
            self.adapter_4_3_batchnorm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_4_4_conv = nn.Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_4_4_conv)
            self.init_bnorm(self.adapter_4_3_batchnorm)

        self.encoder_2_8_batchnorm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.encoder_2_9_relu = nn.ReLU(inplace=True)
        self.encoder_2_10_conv = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        #adapter
        if (self.adapters):
            self.adapter_5_1_batchnorm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_5_2_conv = nn.Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_5_2_conv)
            self.init_bnorm(self.adapter_5_1_batchnorm)
            self.adapter_5_3_batchnorm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_5_4_conv = nn.Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_5_4_conv)
            self.init_bnorm(self.adapter_5_3_batchnorm)

        self.encoder_2_11_batchnorm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.encoder_2_12_relu = nn.ReLU(inplace=True)

        #third block
        self.encoder_3_1_conv = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

        #adapter
        if (self.adapters):
            self.adapter_6_1_batchnorm = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_6_2_conv = nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_6_2_conv)
            self.init_bnorm(self.adapter_6_1_batchnorm)
            self.adapter_6_3_batchnorm = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_6_4_conv = nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_6_4_conv)
            self.init_bnorm(self.adapter_6_3_batchnorm)

        self.encoder_3_2_batchnorm = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.encoder_3_3_relu = nn.ReLU(inplace=True)
        self.encoder_3_4_conv = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        #adapter
        if (self.adapters):
            self.adapter_7_1_batchnorm = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_7_2_conv = nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_7_2_conv)
            self.init_bnorm(self.adapter_7_1_batchnorm)
            self.adapter_7_3_batchnorm = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_7_4_conv = nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_7_4_conv)
            self.init_bnorm(self.adapter_7_3_batchnorm)

        self.encoder_3_5_batchnorm = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        #need it for concatenation
        self.encoder_3_6_downsample = nn.Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
        self.encoder_3_7_batchnorm = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.encoder_3_8_relu = nn.ReLU(inplace=True)

        self.encoder_3_9_conv = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        #adapter
        if (self.adapters):
            self.adapter_8_1_batchnorm = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_8_2_conv = nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_8_2_conv)
            self.init_bnorm(self.adapter_8_1_batchnorm)
            self.adapter_8_3_batchnorm = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_8_4_conv = nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_8_4_conv)
            self.init_bnorm(self.adapter_8_3_batchnorm)

        self.encoder_3_10_batchnorm = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.encoder_3_11_relu = nn.ReLU(inplace=True)
        self.encoder_3_12_conv = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        #adapter
        if (self.adapters):
            self.adapter_9_1_batchnorm = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_9_2_conv = nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_9_2_conv)
            self.init_bnorm(self.adapter_9_1_batchnorm)
            self.adapter_9_3_batchnorm = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_9_4_conv = nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_9_4_conv)
            self.init_bnorm(self.adapter_9_3_batchnorm)

        self.encoder_3_13_batchnorm = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.encoder_3_14_relu = nn.ReLU(inplace=True)

        #fourth block
        self.encoder_4_1_conv = nn.Conv2d(128,256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

        #adapter
        if (self.adapters):
            self.adapter_10_1_batchnorm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_10_2_conv = nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_10_2_conv)
            self.init_bnorm(self.adapter_10_1_batchnorm)
            self.adapter_10_3_batchnorm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_10_4_conv = nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_10_4_conv)
            self.init_bnorm(self.adapter_10_3_batchnorm)

        self.encoder_4_2_batchnorm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.encoder_4_3_relu = nn.ReLU(inplace=True)
        self.encoder_4_4_conv = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        #adapter
        if (self.adapters):
            self.adapter_11_1_batchnorm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_11_2_conv = nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_11_2_conv)
            self.init_bnorm(self.adapter_11_1_batchnorm)
            self.adapter_11_3_batchnorm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_11_4_conv = nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_11_4_conv)
            self.init_bnorm(self.adapter_11_3_batchnorm)

        self.encoder_4_5_batchnorm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        #need it for concatenation
        self.encoder_4_6_downsample = nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
        self.encoder_4_7_batchnorm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.encoder_4_8_relu = nn.ReLU(inplace=True)

        self.encoder_4_9_conv = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        #adapter
        if (self.adapters):
            self.adapter_12_1_batchnorm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_12_2_conv = nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_12_2_conv)
            self.init_bnorm(self.adapter_12_1_batchnorm)
            self.adapter_12_3_batchnorm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_12_4_conv = nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_12_4_conv)
            self.init_bnorm(self.adapter_12_3_batchnorm)

        self.encoder_4_10_batchnorm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.encoder_4_11_relu = nn.ReLU(inplace=True)
        self.encoder_4_12_conv = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        #adapter
        if (self.adapters):
            self.adapter_13_1_batchnorm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_13_2_conv = nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_13_2_conv)
            self.init_bnorm(self.adapter_13_1_batchnorm)
            self.adapter_13_3_batchnorm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_13_4_conv = nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_13_4_conv)
            self.init_bnorm(self.adapter_13_3_batchnorm)

        self.encoder_4_13_batchnorm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.encoder_4_14_relu = nn.ReLU(inplace=True)

        #fifth block
        self.encoder_5_1_conv = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

        #adapter
        if (self.adapters):
            self.adapter_14_1_batchnorm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_14_2_conv = nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_14_2_conv)
            self.init_bnorm(self.adapter_14_1_batchnorm)
            self.adapter_14_3_batchnorm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_14_4_conv = nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_14_4_conv)
            self.init_bnorm(self.adapter_14_3_batchnorm)

        self.encoder_5_2_batchnorm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.encoder_5_3_relu = nn.ReLU(inplace=True)
        self.encoder_5_4_conv = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        #adapter
        if (self.adapters):
            self.adapter_15_1_batchnorm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_15_2_conv = nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_15_2_conv)
            self.init_bnorm(self.adapter_15_1_batchnorm)
            self.adapter_15_3_batchnorm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_15_4_conv = nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_15_4_conv)
            self.init_bnorm(self.adapter_15_3_batchnorm)

        self.encoder_5_5_batchnorm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        #need it for concatenation
        self.encoder_5_6_downsample = nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
        self.encoder_5_7_batchnorm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.encoder_5_8_relu = nn.ReLU(inplace=True)

        self.encoder_5_9_conv = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        #adapter
        if (self.adapters):
            self.adapter_16_1_batchnorm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_16_2_conv = nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_16_2_conv)
            self.init_bnorm(self.adapter_16_1_batchnorm)
            self.adapter_16_3_batchnorm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_16_4_conv = nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_16_4_conv)
            self.init_bnorm(self.adapter_16_3_batchnorm)

        self.encoder_5_10_batchnorm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.encoder_5_11_relu = nn.ReLU(inplace=True)
        self.encoder_5_12_conv = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        #adapter
        if (self.adapters):
            self.adapter_17_1_batchnorm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_17_2_conv = nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_17_2_conv)
            self.init_bnorm(self.adapter_17_1_batchnorm)
            self.adapter_17_3_batchnorm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_17_4_conv = nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_17_4_conv)
            self.init_bnorm(self.adapter_17_3_batchnorm)

        self.encoder_5_13_batchnorm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.encoder_5_14_relu = nn.ReLU(inplace=True)

        #----------------------------------------------------------------        
        #decoder layers (decoder_block_layernumber_typeoflayer)
        #----------------------------------------------------------------

        #first decoder block
        self.decoder_1_1_conv = nn.Conv2d(768, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        #adapter
        if (self.adapters):
            self.adapter_18_1_batchnorm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_18_2_conv = nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_18_2_conv)
            self.init_bnorm(self.adapter_18_1_batchnorm)
            self.adapter_18_3_batchnorm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_18_4_conv = nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_18_4_conv)
            self.init_bnorm(self.adapter_18_3_batchnorm)

        self.decoder_1_2_batchnorm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.decoder_1_3_relu = nn.ReLU(inplace=True)
        self.decoder_1_4_conv = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        
        #adapter
        if (self.adapters):
            self.adapter_19_1_batchnorm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_19_2_conv = nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_19_2_conv)
            self.init_bnorm(self.adapter_19_1_batchnorm)
            self.adapter_19_3_batchnorm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_19_4_conv = nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_19_4_conv)
            self.init_bnorm(self.adapter_19_3_batchnorm)
        
        self.decoder_1_5_batchnorm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.decoder_1_6_relu = nn.ReLU(inplace=True)

        #second decoder block
        self.decoder_2_1_conv = nn.Conv2d(384, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        #adapter
        if (self.adapters):
            self.adapter_20_1_batchnorm = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_20_2_conv = nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_20_2_conv)
            self.init_bnorm(self.adapter_20_1_batchnorm)
            self.adapter_20_3_batchnorm = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_20_4_conv = nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_20_4_conv)
            self.init_bnorm(self.adapter_20_3_batchnorm)

        self.decoder_2_2_batchnorm = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.decoder_2_3_relu = nn.ReLU(inplace=True)
        self.decoder_2_4_conv = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        #adapter
        if (self.adapters):
            self.adapter_21_1_batchnorm = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_21_2_conv = nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_21_2_conv)
            self.init_bnorm(self.adapter_21_1_batchnorm)
            self.adapter_21_3_batchnorm = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_21_4_conv = nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_21_4_conv)
            self.init_bnorm(self.adapter_21_3_batchnorm)

        self.decoder_2_5_batchnorm = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.decoder_2_6_relu = nn.ReLU(inplace=True)

        #third decoder block
        self.decoder_3_1_conv = nn.Conv2d(192, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        #adapter
        if (self.adapters):
            self.adapter_22_1_batchnorm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_22_2_conv = nn.Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_22_2_conv)
            self.init_bnorm(self.adapter_22_1_batchnorm)
            self.adapter_22_3_batchnorm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_22_4_conv = nn.Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_22_4_conv)
            self.init_bnorm(self.adapter_22_3_batchnorm)

        self.decoder_3_2_batchnorm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.decoder_3_3_relu = nn.ReLU(inplace=True)
        self.decoder_3_4_conv = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        #adapter
        if (self.adapters):
            self.adapter_23_1_batchnorm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_23_2_conv = nn.Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_23_2_conv)
            self.init_bnorm(self.adapter_23_1_batchnorm)
            self.adapter_23_3_batchnorm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_23_4_conv = nn.Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_23_4_conv)
            self.init_bnorm(self.adapter_23_3_batchnorm)

        self.decoder_3_5_batchnorm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.decoder_3_6_relu = nn.ReLU(inplace=True)

        #fourth decoder block
        self.decoder_4_1_conv = nn.Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        #adapter
        if (self.adapters):
            self.adapter_24_1_batchnorm = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_24_2_conv = nn.Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_24_2_conv)
            self.init_bnorm(self.adapter_24_1_batchnorm)
            self.adapter_24_3_batchnorm = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_24_4_conv = nn.Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_24_4_conv)
            self.init_bnorm(self.adapter_24_3_batchnorm)

        self.decoder_4_2_batchnorm = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.decoder_4_3_relu = nn.ReLU(inplace=True)
        self.decoder_4_4_conv = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        #adapter
        if (self.adapters):
            self.adapter_25_1_batchnorm = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_25_2_conv = nn.Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_25_2_conv)
            self.init_bnorm(self.adapter_25_1_batchnorm)
            self.adapter_25_3_batchnorm = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_25_4_conv = nn.Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_25_4_conv)
            self.init_bnorm(self.adapter_25_3_batchnorm)

        self.decoder_4_5_batchnorm = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.decoder_4_6_relu = nn.ReLU(inplace=True)

        #fifth decoder block
        self.decoder_5_1_conv = nn.Conv2d(32, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        #adapter
        if (self.adapters):
            self.adapter_26_1_batchnorm = nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_26_2_conv = nn.Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_26_2_conv)
            self.init_bnorm(self.adapter_26_1_batchnorm)
            self.adapter_26_3_batchnorm = nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_26_4_conv = nn.Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_26_4_conv)
            self.init_bnorm(self.adapter_26_3_batchnorm)

        self.decoder_5_2_batchnorm = nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.decoder_5_3_relu = nn.ReLU(inplace=True)
        self.decoder_5_4_conv = nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        #adapter
        if (self.adapters):
            self.adapter_27_1_batchnorm = nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_27_2_conv = nn.Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_27_2_conv)
            self.init_bnorm(self.adapter_27_1_batchnorm)
            self.adapter_27_3_batchnorm = nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_27_4_conv = nn.Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_27_4_conv)
            self.init_bnorm(self.adapter_27_3_batchnorm)

        self.decoder_5_5_batchnorm = nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.decoder_5_6_relu = nn.ReLU(inplace=True)

        #final segmentation head
        self.decoder_6_1_conv = nn.Conv2d(16, num_classes, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    def init_convweights(self, layer):
        layer.weight.data.zero_()
        return

    def init_bnorm(self, layer):
        layer.weight.data.fill_(1)
        layer.bias.data.zero_()
        layer.running_mean.zero_()
        layer.running_var.fill_(1)
        layer.num_batches_tracked.zero_()
        return 
    
    def forward(self, x):

        # print("Input Shape:")
        # print(x.size())

        #first block
        block1 = self.encoder_1_1_conv(x)

        if (self.adapters):
            adapt1 = self.adapter_1_1_batchnorm(block1)
            adapt1 = self.adapter_1_2_conv(adapt1)
            adapt1 = self.adapter_1_3_batchnorm(adapt1)
            adapt1 = self.adapter_1_4_conv(adapt1)
            block1 = block1 + adapt1

        block1 = self.encoder_1_2_batchnorm(block1)
        block1 = self.encoder_1_3_relu(block1)
        x1 = block1 #save it for decoder
        block1 = self.encoder_1_4_maxpool(block1)

        #second block
        block2 = self.encoder_2_1_conv(block1)

        if (self.adapters):
            adapt2 = self.adapter_2_1_batchnorm(block2)
            adapt2 = self.adapter_2_2_conv(adapt2)
            adapt2 = self.adapter_2_3_batchnorm(adapt2)
            adapt2 = self.adapter_2_4_conv(adapt2)
            block2 = block2 + adapt2

        block2 = self.encoder_2_2_batchnorm(block2) 
        block2 = self.encoder_2_3_relu(block2)
        block2 = self.encoder_2_4_conv(block2)

        if (self.adapters):
            adapt3 = self.adapter_3_1_batchnorm(block2)
            adapt3 = self.adapter_3_2_conv(adapt3)
            adapt3 = self.adapter_3_3_batchnorm(adapt3)
            adapt3 = self.adapter_3_4_conv(adapt3)
            block2 = block2 + adapt3

        block2 = self.encoder_2_5_batchnorm(block2)

        #skip connection
        x2 = block1 + block2
        x2 = self.encoder_2_6_relu(x2)

        block2 = self.encoder_2_7_conv(x2)

        if (self.adapters):
            adapt4 = self.adapter_4_1_batchnorm(block2)
            adapt4 = self.adapter_4_2_conv(adapt4)
            adapt4 = self.adapter_4_3_batchnorm(adapt4)
            adapt4 = self.adapter_4_4_conv(adapt4)
            block2 = block2 + adapt4

        block2 = self.encoder_2_8_batchnorm(block2) 
        block2 = self.encoder_2_9_relu(block2)
        block2 = self.encoder_2_10_conv(block2)

        if (self.adapters):
            adapt5 = self.adapter_5_1_batchnorm(block2)
            adapt5 = self.adapter_5_2_conv(adapt5)
            adapt5 = self.adapter_5_3_batchnorm(adapt5)
            adapt5 = self.adapter_5_4_conv(adapt5)
            block2 = block2 + adapt5

        block2 = self.encoder_2_11_batchnorm(block2)

        #skip connection
        x3 = x2 + block2
        x3 = self.encoder_2_12_relu(x3)

        #third block
        block3 = self.encoder_3_1_conv(x3)

        if (self.adapters):
            adapt6 = self.adapter_6_1_batchnorm(block3)
            adapt6 = self.adapter_6_2_conv(adapt6)
            adapt6 = self.adapter_6_3_batchnorm(adapt6)
            adapt6 = self.adapter_6_4_conv(adapt6)
            block3 = block3 + adapt6

        block3 = self.encoder_3_2_batchnorm(block3) 
        block3 = self.encoder_3_3_relu(block3) 
        block3 = self.encoder_3_4_conv(block3) 

        if (self.adapters):
            adapt7 = self.adapter_7_1_batchnorm(block3)
            adapt7 = self.adapter_7_2_conv(adapt7)
            adapt7 = self.adapter_7_3_batchnorm(adapt7)
            adapt7 = self.adapter_7_4_conv(adapt7)
            block3 = block3 + adapt7

        block3 = self.encoder_3_5_batchnorm(block3) 

        #need it for concatenation
        x4 = self.encoder_3_6_downsample(x3) 
        x4 = self.encoder_3_7_batchnorm(x4)
        x4 = x4 + block3
        x4 = self.encoder_3_8_relu(x4) 

        block3 = self.encoder_3_9_conv(x4) 

        if (self.adapters):
            adapt8 = self.adapter_8_1_batchnorm(block3)
            adapt8 = self.adapter_8_2_conv(adapt8)
            adapt8 = self.adapter_8_3_batchnorm(adapt8)
            adapt8 = self.adapter_8_4_conv(adapt8)
            block3 = block3 + adapt8

        block3 = self.encoder_3_10_batchnorm(block3)
        block3 = self.encoder_3_11_relu(block3)
        block3 = self.encoder_3_12_conv(block3)

        if (self.adapters):
            adapt9 = self.adapter_9_1_batchnorm(block3)
            adapt9 = self.adapter_9_2_conv(adapt9)
            adapt9 = self.adapter_9_3_batchnorm(adapt9)
            adapt9 = self.adapter_9_4_conv(adapt9)
            block3 = block3 + adapt9

        block3 = self.encoder_3_13_batchnorm(block3)

        #skip connection
        x5 = x4 + block3
        x5 = self.encoder_3_14_relu(x5)

        #fourth block
        block4 = self.encoder_4_1_conv(x5)

        if (self.adapters):
            adapt10 = self.adapter_10_1_batchnorm(block4)
            adapt10 = self.adapter_10_2_conv(adapt10)
            adapt10 = self.adapter_10_3_batchnorm(adapt10)
            adapt10 = self.adapter_10_4_conv(adapt10)
            block4 = block4 + adapt10

        block4 = self.encoder_4_2_batchnorm(block4) 
        block4 = self.encoder_4_3_relu(block4) 
        block4 = self.encoder_4_4_conv(block4) 

        if (self.adapters):
            adapt11 = self.adapter_11_1_batchnorm(block4)
            adapt11 = self.adapter_11_2_conv(adapt11)
            adapt11 = self.adapter_11_3_batchnorm(adapt11)
            adapt11 = self.adapter_11_4_conv(adapt11)
            block4 = block4 + adapt11

        block4 = self.encoder_4_5_batchnorm(block4) 

        #need it for concatenation
        x6 = self.encoder_4_6_downsample(x5) 
        x6 = self.encoder_4_7_batchnorm(x6)
        x6 = x6 + block4
        x6 = self.encoder_4_8_relu(x6)

        block4 = self.encoder_4_9_conv(x6) 

        if (self.adapters):
            adapt12 = self.adapter_12_1_batchnorm(block4)
            adapt12 = self.adapter_12_2_conv(adapt12)
            adapt12 = self.adapter_12_3_batchnorm(adapt12)
            adapt12 = self.adapter_12_4_conv(adapt12)
            block4 = block4 + adapt12

        block4 = self.encoder_4_10_batchnorm(block4)
        block4 = self.encoder_4_11_relu(block4)
        block4 = self.encoder_4_12_conv(block4)

        if (self.adapters):
            adapt13 = self.adapter_13_1_batchnorm(block4)
            adapt13 = self.adapter_13_2_conv(adapt13)
            adapt13 = self.adapter_13_3_batchnorm(adapt13)
            adapt13 = self.adapter_13_4_conv(adapt13)
            block4 = block4 + adapt13

        block4 = self.encoder_4_13_batchnorm(block4)
        
        #skip connection
        x7 = x6 + block4
        x7 = self.encoder_4_14_relu(x7)

        #fifth block
        block5 = self.encoder_5_1_conv(x7)

        if (self.adapters):
            adapt14 = self.adapter_14_1_batchnorm(block5)
            adapt14 = self.adapter_14_2_conv(adapt14)
            adapt14 = self.adapter_14_3_batchnorm(adapt14)
            adapt14 = self.adapter_14_4_conv(adapt14)
            block5 = block5 + adapt14

        block5 = self.encoder_5_2_batchnorm(block5) 
        block5 = self.encoder_5_3_relu(block5) 
        block5 = self.encoder_5_4_conv(block5) 

        if (self.adapters):
            adapt15 = self.adapter_15_1_batchnorm(block5)
            adapt15 = self.adapter_15_2_conv(adapt15)
            adapt15 = self.adapter_15_3_batchnorm(adapt15)
            adapt15 = self.adapter_15_4_conv(adapt15)
            block5 = block5 + adapt15

        block5 = self.encoder_5_5_batchnorm(block5) 

        #need it for concatenation
        x8 = self.encoder_5_6_downsample(x7) 
        x8 = self.encoder_5_7_batchnorm(x8)
        x8 = x8 + block5
        x8 = self.encoder_5_8_relu(x8)

        block5 = self.encoder_5_9_conv(x8) 

        if (self.adapters):
            adapt16 = self.adapter_16_1_batchnorm(block5)
            adapt16 = self.adapter_16_2_conv(adapt16)
            adapt16 = self.adapter_16_3_batchnorm(adapt16)
            adapt16 = self.adapter_16_4_conv(adapt16)
            block5 = block5 + adapt16

        block5 = self.encoder_5_10_batchnorm(block5)
        block5 = self.encoder_5_11_relu(block5)
        block5 = self.encoder_5_12_conv(block5)

        if (self.adapters):
            adapt17 = self.adapter_17_1_batchnorm(block5)
            adapt17 = self.adapter_17_2_conv(adapt17)
            adapt17 = self.adapter_17_3_batchnorm(adapt17)
            adapt17 = self.adapter_17_4_conv(adapt17)
            block5 = block5 + adapt17

        block5 = self.encoder_5_13_batchnorm(block5)

        #skip connection
        x9 = x8 + block5 #bottleneck
        x9 = self.encoder_5_14_relu(x9)

        # print("Encoder Shapes: ")
        # print(x1.size(),block2.size(),block3.size(),block4.size(),block5.size())

        #upsampling + concatenation
        y1 = F.interpolate(x9, scale_factor=2, mode="nearest")
        y1 = torch.cat([y1,x7],dim=1) #512 + 256

        #first decoder block
        deblock1 = self.decoder_1_1_conv(y1)

        if (self.adapters):
            adapt18 = self.adapter_18_1_batchnorm(deblock1)
            adapt18 = self.adapter_18_2_conv(adapt18)
            adapt18 = self.adapter_18_3_batchnorm(adapt18)
            adapt18 = self.adapter_18_4_conv(adapt18)
            deblock1 = deblock1 + adapt18

        deblock1 = self.decoder_1_2_batchnorm(deblock1)
        deblock1 = self.decoder_1_3_relu(deblock1)
        deblock1 = self.decoder_1_4_conv(deblock1)

        if (self.adapters):
            adapt19 = self.adapter_19_1_batchnorm(deblock1)
            adapt19 = self.adapter_19_2_conv(adapt19)
            adapt19 = self.adapter_19_3_batchnorm(adapt19)
            adapt19 = self.adapter_19_4_conv(adapt19)
            deblock1 = deblock1 + adapt19

        deblock1 = self.decoder_1_5_batchnorm(deblock1)
        deblock1 = self.decoder_1_6_relu(deblock1)

        #upsampling + concatenation
        deblock1 = F.interpolate(deblock1, scale_factor=2, mode="nearest")
        y2 = torch.cat([deblock1,x5],dim=1) #256 + 128

        #second decoder block
        deblock2 = self.decoder_2_1_conv(y2)

        if (self.adapters):
            adapt20 = self.adapter_20_1_batchnorm(deblock2)
            adapt20 = self.adapter_20_2_conv(adapt20)
            adapt20 = self.adapter_20_3_batchnorm(adapt20)
            adapt20 = self.adapter_20_4_conv(adapt20)
            deblock2 = deblock2 + adapt20

        deblock2 = self.decoder_2_2_batchnorm(deblock2)
        deblock2 = self.decoder_2_3_relu(deblock2)
        deblock2 = self.decoder_2_4_conv(deblock2)

        if (self.adapters):
            adapt21 = self.adapter_21_1_batchnorm(deblock2)
            adapt21 = self.adapter_21_2_conv(adapt21)
            adapt21 = self.adapter_21_3_batchnorm(adapt21)
            adapt21 = self.adapter_21_4_conv(adapt21)
            deblock2 = deblock2 + adapt21

        deblock2 = self.decoder_2_5_batchnorm(deblock2)
        deblock2 = self.decoder_2_6_relu(deblock2)

        #upsampling + concatenation
        deblock2 = F.interpolate(deblock2, scale_factor=2, mode="nearest")
        y3 = torch.cat([deblock2,x3],dim=1) #128 + 64

        #third decoder block
        deblock3 = self.decoder_3_1_conv(y3)

        if (self.adapters):
            adapt22 = self.adapter_22_1_batchnorm(deblock3)
            adapt22 = self.adapter_22_2_conv(adapt22)
            adapt22 = self.adapter_22_3_batchnorm(adapt22)
            adapt22 = self.adapter_22_4_conv(adapt22)
            deblock3 = deblock3 + adapt22

        deblock3 = self.decoder_3_2_batchnorm(deblock3)
        deblock3 = self.decoder_3_3_relu(deblock3)
        deblock3 = self.decoder_3_4_conv(deblock3)

        if (self.adapters):
            adapt23 = self.adapter_23_1_batchnorm(deblock3)
            adapt23 = self.adapter_23_2_conv(adapt23)
            adapt23 = self.adapter_23_3_batchnorm(adapt23)
            adapt23 = self.adapter_23_4_conv(adapt23)
            deblock3 = deblock3 + adapt23

        deblock3 = self.decoder_3_5_batchnorm(deblock3)
        deblock3 = self.decoder_3_6_relu(deblock3)

        #upsampling + concatenation
        deblock3 = F.interpolate(deblock3, scale_factor=2, mode="nearest")
        y4 = torch.cat([deblock3,x1],dim=1) #64 + 64

        #fourth decoder block
        deblock4 = self.decoder_4_1_conv(y4)

        if (self.adapters):
            adapt24 = self.adapter_24_1_batchnorm(deblock4)
            adapt24 = self.adapter_24_2_conv(adapt24)
            adapt24 = self.adapter_24_3_batchnorm(adapt24)
            adapt24 = self.adapter_24_4_conv(adapt24)
            deblock4 = deblock4 + adapt24

        deblock4 = self.decoder_4_2_batchnorm(deblock4)
        deblock4 = self.decoder_4_3_relu(deblock4)
        deblock4 = self.decoder_4_4_conv(deblock4)

        if (self.adapters):
            adapt25 = self.adapter_25_1_batchnorm(deblock4)
            adapt25 = self.adapter_25_2_conv(adapt25)
            adapt25 = self.adapter_25_3_batchnorm(adapt25)
            adapt25 = self.adapter_25_4_conv(adapt25)
            deblock4 = deblock4 + adapt25

        deblock4 = self.decoder_4_5_batchnorm(deblock4)
        deblock4 = self.decoder_4_6_relu(deblock4)

        #no concatenation
        deblock4 = F.interpolate(deblock4, scale_factor=2, mode="nearest")
        y5 = deblock4 #32

        #fifth decoder block (segmentation head)
        deblock5 = self.decoder_5_1_conv(y5)

        if (self.adapters):
            adapt26 = self.adapter_26_1_batchnorm(deblock5)
            adapt26 = self.adapter_26_2_conv(adapt26)
            adapt26 = self.adapter_26_3_batchnorm(adapt26)
            adapt26 = self.adapter_26_4_conv(adapt26)
            deblock5 = deblock5 + adapt26

        deblock5 = self.decoder_5_2_batchnorm(deblock5)
        deblock5 = self.decoder_5_3_relu(deblock5)
        deblock5 = self.decoder_5_4_conv(deblock5)

        if (self.adapters):
            adapt27 = self.adapter_27_1_batchnorm(deblock5)
            adapt27 = self.adapter_27_2_conv(adapt27)
            adapt27 = self.adapter_27_3_batchnorm(adapt27)
            adapt27 = self.adapter_27_4_conv(adapt27)
            deblock5 = deblock5 + adapt27

        deblock5 = self.decoder_5_5_batchnorm(deblock5)
        deblock5 = self.decoder_5_6_relu(deblock5)

        #no concatenation (no upsampling in seg head)
        #deblock5 = F.interpolate(deblock5, scale_factor=2, mode="nearest")
        y6 = deblock5 #16

        #sixth decoder block
        deblock6 = self.decoder_6_1_conv(y6)

        # print("Decoder Shapes:")
        # print(y1.size(),y2.size(),y3.size(),y4.size(),y5.size(),y6.size(),deblock6.size())

        return deblock6

#resnet18 (fuse)
class Unet_18_fuse(nn.Module):

    def __init__(self, in_channels, num_classes):

        super(Unet_18_fuse, self).__init__()

        #----------------------------------------------------------------        
        #encoder layers (encoder_block_layernumber_typeoflayer)
        #----------------------------------------------------------------

        #first block
        self.encoder_1_1_conv = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)   
        self.encoder_1_3_relu = nn.ReLU(inplace=True)
        self.encoder_1_4_maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        #second block
        self.encoder_2_1_conv = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.encoder_2_3_relu = nn.ReLU(inplace=True)
        self.encoder_2_4_conv = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.encoder_2_6_relu = nn.ReLU(inplace=True)
        self.encoder_2_7_conv = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.encoder_2_9_relu = nn.ReLU(inplace=True)
        self.encoder_2_10_conv = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.encoder_2_12_relu = nn.ReLU(inplace=True)

        #third block
        self.encoder_3_1_conv = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.encoder_3_3_relu = nn.ReLU(inplace=True)
        self.encoder_3_4_conv = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        #need it for concatenation
        self.encoder_3_6_downsample = nn.Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
        self.encoder_3_7_batchnorm = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.encoder_3_8_relu = nn.ReLU(inplace=True)

        self.encoder_3_9_conv = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.encoder_3_11_relu = nn.ReLU(inplace=True)
        self.encoder_3_12_conv = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.encoder_3_14_relu = nn.ReLU(inplace=True)

        #fourth block
        self.encoder_4_1_conv = nn.Conv2d(128,256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.encoder_4_3_relu = nn.ReLU(inplace=True)
        self.encoder_4_4_conv = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        #need it for concatenation
        self.encoder_4_6_downsample = nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
        self.encoder_4_7_batchnorm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.encoder_4_8_relu = nn.ReLU(inplace=True)

        self.encoder_4_9_conv = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.encoder_4_11_relu = nn.ReLU(inplace=True)
        self.encoder_4_12_conv = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.encoder_4_14_relu = nn.ReLU(inplace=True)

        #fifth block
        self.encoder_5_1_conv = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.encoder_5_3_relu = nn.ReLU(inplace=True)
        self.encoder_5_4_conv = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        #need it for concatenation
        self.encoder_5_6_downsample = nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
        self.encoder_5_7_batchnorm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.encoder_5_8_relu = nn.ReLU(inplace=True)

        self.encoder_5_9_conv = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.encoder_5_11_relu = nn.ReLU(inplace=True)
        self.encoder_5_12_conv = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.encoder_5_14_relu = nn.ReLU(inplace=True)

        #----------------------------------------------------------------        
        #decoder layers (decoder_block_layernumber_typeoflayer)
        #----------------------------------------------------------------

        #first decoder block
        self.decoder_1_1_conv = nn.Conv2d(768, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.decoder_1_3_relu = nn.ReLU(inplace=True)
        self.decoder_1_4_conv = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.decoder_1_6_relu = nn.ReLU(inplace=True)

        #second decoder block
        self.decoder_2_1_conv = nn.Conv2d(384, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.decoder_2_3_relu = nn.ReLU(inplace=True)
        self.decoder_2_4_conv = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.decoder_2_6_relu = nn.ReLU(inplace=True)

        #third decoder block
        self.decoder_3_1_conv = nn.Conv2d(192, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.decoder_3_3_relu = nn.ReLU(inplace=True)
        self.decoder_3_4_conv = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.decoder_3_6_relu = nn.ReLU(inplace=True)

        #fourth decoder block
        self.decoder_4_1_conv = nn.Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.decoder_4_3_relu = nn.ReLU(inplace=True)
        self.decoder_4_4_conv = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.decoder_4_6_relu = nn.ReLU(inplace=True)

        #fifth decoder block
        self.decoder_5_1_conv = nn.Conv2d(32, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.decoder_5_3_relu = nn.ReLU(inplace=True)
        self.decoder_5_4_conv = nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.decoder_5_6_relu = nn.ReLU(inplace=True)

        #final segmentation head
        self.decoder_6_1_conv = nn.Conv2d(16, num_classes, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    
    def forward(self, x):

        #first block
        block1 = self.encoder_1_1_conv(x)
        block1 = self.encoder_1_3_relu(block1)
        x1 = block1 #save it for decoder
        block1 = self.encoder_1_4_maxpool(block1)

        #second block
        block2 = self.encoder_2_1_conv(block1)
        block2 = self.encoder_2_3_relu(block2)
        block2 = self.encoder_2_4_conv(block2)

        #skip connection
        x2 = block1 + block2
        x2 = self.encoder_2_6_relu(x2)

        block2 = self.encoder_2_7_conv(x2)
        block2 = self.encoder_2_9_relu(block2)
        block2 = self.encoder_2_10_conv(block2)

        #skip connection
        x3 = x2 + block2
        x3 = self.encoder_2_12_relu(x3)

        #third block
        block3 = self.encoder_3_1_conv(x3)
        block3 = self.encoder_3_3_relu(block3) 
        block3 = self.encoder_3_4_conv(block3) 

        #need it for concatenation
        x4 = self.encoder_3_6_downsample(x3) 
        x4 = self.encoder_3_7_batchnorm(x4)
        x4 = x4 + block3
        x4 = self.encoder_3_8_relu(x4) 

        block3 = self.encoder_3_9_conv(x4) 
        block3 = self.encoder_3_11_relu(block3)
        block3 = self.encoder_3_12_conv(block3)

        #skip connection
        x5 = x4 + block3
        x5 = self.encoder_3_14_relu(x5)

        #fourth block
        block4 = self.encoder_4_1_conv(x5)
        block4 = self.encoder_4_3_relu(block4) 
        block4 = self.encoder_4_4_conv(block4) 

        #need it for concatenation
        x6 = self.encoder_4_6_downsample(x5) 
        x6 = self.encoder_4_7_batchnorm(x6)
        x6 = x6 + block4
        x6 = self.encoder_4_8_relu(x6)

        block4 = self.encoder_4_9_conv(x6) 
        block4 = self.encoder_4_11_relu(block4)
        block4 = self.encoder_4_12_conv(block4)
        
        #skip connection
        x7 = x6 + block4
        x7 = self.encoder_4_14_relu(x7)

        #fifth block
        block5 = self.encoder_5_1_conv(x7) 
        block5 = self.encoder_5_3_relu(block5) 
        block5 = self.encoder_5_4_conv(block5) 

        #need it for concatenation
        x8 = self.encoder_5_6_downsample(x7) 
        x8 = self.encoder_5_7_batchnorm(x8)
        x8 = x8 + block5
        x8 = self.encoder_5_8_relu(x8)

        block5 = self.encoder_5_9_conv(x8) 
        block5 = self.encoder_5_11_relu(block5)
        block5 = self.encoder_5_12_conv(block5)

        #skip connection
        x9 = x8 + block5 #bottleneck
        x9 = self.encoder_5_14_relu(x9)

        # print("Encoder Shapes: ")
        # print(x1.size(),block2.size(),block3.size(),block4.size(),block5.size())

        #upsampling + concatenation
        y1 = F.interpolate(x9, scale_factor=2, mode="nearest")
        y1 = torch.cat([y1,x7],dim=1) #512 + 256

        #first decoder block
        deblock1 = self.decoder_1_1_conv(y1)
        deblock1 = self.decoder_1_3_relu(deblock1)
        deblock1 = self.decoder_1_4_conv(deblock1)
        deblock1 = self.decoder_1_6_relu(deblock1)

        #upsampling + concatenation
        deblock1 = F.interpolate(deblock1, scale_factor=2, mode="nearest")
        y2 = torch.cat([deblock1,x5],dim=1) #256 + 128

        #second decoder block
        deblock2 = self.decoder_2_1_conv(y2)
        deblock2 = self.decoder_2_3_relu(deblock2)
        deblock2 = self.decoder_2_4_conv(deblock2)
        deblock2 = self.decoder_2_6_relu(deblock2)

        #upsampling + concatenation
        deblock2 = F.interpolate(deblock2, scale_factor=2, mode="nearest")
        y3 = torch.cat([deblock2,x3],dim=1) #128 + 64

        #third decoder block
        deblock3 = self.decoder_3_1_conv(y3)
        deblock3 = self.decoder_3_3_relu(deblock3)
        deblock3 = self.decoder_3_4_conv(deblock3)
        deblock3 = self.decoder_3_6_relu(deblock3)

        #upsampling + concatenation
        deblock3 = F.interpolate(deblock3, scale_factor=2, mode="nearest")
        y4 = torch.cat([deblock3,x1],dim=1) #64 + 64

        #fourth decoder block
        deblock4 = self.decoder_4_1_conv(y4)
        deblock4 = self.decoder_4_3_relu(deblock4)
        deblock4 = self.decoder_4_4_conv(deblock4)
        deblock4 = self.decoder_4_6_relu(deblock4)

        #no concatenation
        deblock4 = F.interpolate(deblock4, scale_factor=2, mode="nearest")
        y5 = deblock4 #32

        #fifth decoder block (segmentation head)
        deblock5 = self.decoder_5_1_conv(y5)
        deblock5 = self.decoder_5_3_relu(deblock5)
        deblock5 = self.decoder_5_4_conv(deblock5)
        deblock5 = self.decoder_5_6_relu(deblock5)

        #no concatenation (no upsampling in seg head)
        y6 = deblock5 #16

        #sixth decoder block
        deblock6 = self.decoder_6_1_conv(y6)

        return deblock6

#vgg19_bn
class Unet_vgg19(nn.Module):

    def __init__(self, in_channels, num_classes, adapters):

        super(Unet_vgg19, self).__init__()
        self.adapters = adapters

        #----------------------------------------------------------------        
        #encoder layers (encoder_block_layernumber_typeoflayer)
        #----------------------------------------------------------------

        #first block
        self.encoder_1_1_conv = nn.Conv2d(in_channels, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.encoder_1_2_batchnorm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.encoder_1_3_relu = nn.ReLU(inplace=True)
        self.encoder_1_4_conv = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) 
        self.encoder_1_5_batchnorm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.encoder_1_6_relu = nn.ReLU(inplace=True)
        self.encoder_1_7_maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

        if (self.adapters):
            self.adapter_1_1_batchnorm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_1_2_conv = nn.Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_1_2_conv)
            self.init_bnorm(self.adapter_1_1_batchnorm)
            self.adapter_2_1_batchnorm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_2_2_conv = nn.Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_2_2_conv)
            self.init_bnorm(self.adapter_2_1_batchnorm)

        #second block
        self.encoder_2_1_conv = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.encoder_2_2_batchnorm = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.encoder_2_3_relu = nn.ReLU(inplace=True)
        self.encoder_2_4_conv = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.encoder_2_5_batchnorm = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.encoder_2_6_relu = nn.ReLU(inplace=True)
        self.encoder_2_7_maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

        if (self.adapters):
            self.adapter_3_1_batchnorm = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_3_2_conv = nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_3_2_conv)
            self.init_bnorm(self.adapter_3_1_batchnorm)
            self.adapter_4_1_batchnorm = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_4_2_conv = nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_4_2_conv)
            self.init_bnorm(self.adapter_4_1_batchnorm)
      
        #third block
        self.encoder_3_1_conv = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.encoder_3_2_batchnorm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.encoder_3_3_relu = nn.ReLU(inplace=True)
        self.encoder_3_4_conv = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.encoder_3_5_batchnorm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.encoder_3_6_relu = nn.ReLU(inplace=True)
        self.encoder_3_7_conv = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.encoder_3_8_batchnorm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.encoder_3_9_relu = nn.ReLU(inplace=True)
        self.encoder_3_10_conv = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.encoder_3_11_batchnorm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.encoder_3_12_relu = nn.ReLU(inplace=True)
        self.encoder_3_13_maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

        if (self.adapters):
            self.adapter_5_1_batchnorm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_5_2_conv = nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_5_2_conv)
            self.init_bnorm(self.adapter_5_1_batchnorm)
            self.adapter_6_1_batchnorm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_6_2_conv = nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_6_2_conv)
            self.init_bnorm(self.adapter_6_1_batchnorm)
            self.adapter_7_1_batchnorm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_7_2_conv = nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_7_2_conv)
            self.init_bnorm(self.adapter_7_1_batchnorm)
            self.adapter_8_1_batchnorm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_8_2_conv = nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_8_2_conv)
            self.init_bnorm(self.adapter_8_1_batchnorm)
      
        #fourth block
        self.encoder_4_1_conv = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.encoder_4_2_batchnorm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.encoder_4_3_relu = nn.ReLU(inplace=True)
        self.encoder_4_4_conv = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.encoder_4_5_batchnorm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.encoder_4_6_relu = nn.ReLU(inplace=True)
        self.encoder_4_7_conv = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.encoder_4_8_batchnorm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.encoder_4_9_relu = nn.ReLU(inplace=True)
        self.encoder_4_10_conv = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.encoder_4_11_batchnorm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.encoder_4_12_relu = nn.ReLU(inplace=True)
        self.encoder_4_13_maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

        if (self.adapters):
            self.adapter_9_1_batchnorm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_9_2_conv = nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_9_2_conv)
            self.init_bnorm(self.adapter_9_1_batchnorm)
            self.adapter_10_1_batchnorm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_10_2_conv = nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_10_2_conv)
            self.init_bnorm(self.adapter_10_1_batchnorm)
            self.adapter_11_1_batchnorm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_11_2_conv = nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_11_2_conv)
            self.init_bnorm(self.adapter_11_1_batchnorm)
            self.adapter_12_1_batchnorm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_12_2_conv = nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_12_2_conv)
            self.init_bnorm(self.adapter_12_1_batchnorm)
    
        #fifth block
        self.encoder_5_1_conv = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.encoder_5_2_batchnorm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.encoder_5_3_relu = nn.ReLU(inplace=True)
        self.encoder_5_4_conv = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.encoder_5_5_batchnorm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.encoder_5_6_relu = nn.ReLU(inplace=True)
        self.encoder_5_7_conv = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.encoder_5_8_batchnorm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.encoder_5_9_relu = nn.ReLU(inplace=True)
        self.encoder_5_10_conv = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.encoder_5_11_batchnorm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.encoder_5_12_relu = nn.ReLU(inplace=True)
        self.encoder_5_13_maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.encoder_5_14_avgpool = nn.AdaptiveAvgPool2d(output_size=(7, 7))

        if (self.adapters):
            self.adapter_13_1_batchnorm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_13_2_conv = nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_13_2_conv)
            self.init_bnorm(self.adapter_13_1_batchnorm)
            self.adapter_14_1_batchnorm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_14_2_conv = nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_14_2_conv)
            self.init_bnorm(self.adapter_14_1_batchnorm)
            self.adapter_15_1_batchnorm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_15_2_conv = nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_15_2_conv)
            self.init_bnorm(self.adapter_15_1_batchnorm)
            self.adapter_16_1_batchnorm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_16_2_conv = nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_16_2_conv)
            self.init_bnorm(self.adapter_16_1_batchnorm)

        #----------------------------------------------------------------        
        #decoder layers (decoder_block_layernumber_typeoflayer)
        #----------------------------------------------------------------

        #center block
        self.decoder_1_1_conv = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.decoder_1_2_batchnorm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.decoder_1_3_relu = nn.ReLU(inplace=True)
        self.decoder_1_4_conv = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.decoder_1_5_batchnorm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.decoder_1_6_relu = nn.ReLU(inplace=True)

        if (self.adapters):
            self.adapter_17_1_batchnorm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_17_2_conv = nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_17_2_conv)
            self.init_bnorm(self.adapter_17_1_batchnorm)
            self.adapter_18_1_batchnorm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_18_2_conv = nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_18_2_conv)
            self.init_bnorm(self.adapter_18_1_batchnorm)

        #first decoder block
        self.decoder_2_1_conv = nn.Conv2d(1024, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.decoder_2_2_batchnorm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.decoder_2_3_relu = nn.ReLU(inplace=True)
        self.decoder_2_4_conv = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.decoder_2_5_batchnorm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.decoder_2_6_relu = nn.ReLU(inplace=True)

        if (self.adapters):
            self.adapter_19_1_batchnorm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_19_2_conv = nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_19_2_conv)
            self.init_bnorm(self.adapter_19_1_batchnorm)
            self.adapter_20_1_batchnorm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_20_2_conv = nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_20_2_conv)
            self.init_bnorm(self.adapter_20_1_batchnorm)

        #second decoder block
        self.decoder_3_1_conv = nn.Conv2d(768, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.decoder_3_2_batchnorm = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.decoder_3_3_relu = nn.ReLU(inplace=True)
        self.decoder_3_4_conv = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.decoder_3_5_batchnorm = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.decoder_3_6_relu = nn.ReLU(inplace=True)

        if (self.adapters):
            self.adapter_21_1_batchnorm = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_21_2_conv = nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_21_2_conv)
            self.init_bnorm(self.adapter_21_1_batchnorm)
            self.adapter_22_1_batchnorm = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_22_2_conv = nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_22_2_conv)
            self.init_bnorm(self.adapter_22_1_batchnorm)

        #third decoder block
        self.decoder_4_1_conv = nn.Conv2d(384, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.decoder_4_2_batchnorm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.decoder_4_3_relu = nn.ReLU(inplace=True)
        self.decoder_4_4_conv = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.decoder_4_5_batchnorm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.decoder_4_6_relu = nn.ReLU(inplace=True)

        if (self.adapters):
            self.adapter_23_1_batchnorm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_23_2_conv = nn.Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_23_2_conv)
            self.init_bnorm(self.adapter_23_1_batchnorm)
            self.adapter_24_1_batchnorm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_24_2_conv = nn.Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_24_2_conv)
            self.init_bnorm(self.adapter_24_1_batchnorm)

        #fourth decoder block
        self.decoder_5_1_conv = nn.Conv2d(192, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.decoder_5_2_batchnorm = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.decoder_5_3_relu = nn.ReLU(inplace=True)
        self.decoder_5_4_conv = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.decoder_5_5_batchnorm = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.decoder_5_6_relu = nn.ReLU(inplace=True)

        if (self.adapters):
            self.adapter_25_1_batchnorm = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_25_2_conv = nn.Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_25_2_conv)
            self.init_bnorm(self.adapter_25_1_batchnorm)
            self.adapter_26_1_batchnorm = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_26_2_conv = nn.Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_26_2_conv)
            self.init_bnorm(self.adapter_26_1_batchnorm)

        #fifth decoder block
        self.decoder_6_1_conv = nn.Conv2d(32, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.decoder_6_2_batchnorm = nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.decoder_6_3_relu = nn.ReLU(inplace=True)
        self.decoder_6_4_conv = nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.decoder_6_5_batchnorm = nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.decoder_6_6_relu = nn.ReLU(inplace=True)

        if (self.adapters):
            self.adapter_27_1_batchnorm = nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_27_2_conv = nn.Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_27_2_conv)
            self.init_bnorm(self.adapter_27_1_batchnorm)
            self.adapter_28_1_batchnorm = nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_28_2_conv = nn.Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_28_2_conv)
            self.init_bnorm(self.adapter_28_1_batchnorm)

        #final segmentation head
        self.decoder_7_1_conv = nn.Conv2d(16, num_classes, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    def init_convweights(self, layer):
        layer.weight.data.zero_()
        return

    def init_bnorm(self, layer):
        layer.weight.data.fill_(1)
        layer.bias.data.zero_()
        layer.running_mean.zero_()
        layer.running_var.fill_(1)
        layer.num_batches_tracked.zero_()
        return
    
    def forward(self, x):

        # print("Input Shape:")
        # print(x.size())

        # if (self.adapters):
        #     adapt1 = self.adapter_1_1_batchnorm(block1)
        #     adapt1 = self.adapter_1_2_conv(adapt1)
        #     block1 = block1 + adapt1

        #first block
        block1 = self.encoder_1_1_conv(x)
        if (self.adapters):
            adapt1 = self.adapter_1_1_batchnorm(block1)
            adapt1 = self.adapter_1_2_conv(adapt1)
            block1 = block1 + adapt1
        block1 = self.encoder_1_2_batchnorm(block1)
        block1 = self.encoder_1_3_relu(block1)
        block1 = self.encoder_1_4_conv(block1)
        if (self.adapters):
            adapt2 = self.adapter_2_1_batchnorm(block1)
            adapt2 = self.adapter_2_2_conv(adapt2)
            block1 = block1 + adapt2
        block1 = self.encoder_1_5_batchnorm(block1)
        block1 = self.encoder_1_6_relu(block1)
        x1 = block1 #save it for decoder
        block1 = self.encoder_1_7_maxpool(block1)

        #second block
        block2 = self.encoder_2_1_conv(block1)
        if (self.adapters):
            adapt3 = self.adapter_3_1_batchnorm(block2)
            adapt3 = self.adapter_3_2_conv(adapt3)
            block2 = block2 + adapt3
        block2 = self.encoder_2_2_batchnorm(block2) 
        block2 = self.encoder_2_3_relu(block2)
        block2 = self.encoder_2_4_conv(block2)
        if (self.adapters):
            adapt4 = self.adapter_4_1_batchnorm(block2)
            adapt4 = self.adapter_4_2_conv(adapt4)
            block2 = block2 + adapt4
        block2 = self.encoder_2_5_batchnorm(block2)
        block2 = self.encoder_2_6_relu(block2)
        x2 = block2 #save it for decoder
        block2 = self.encoder_2_7_maxpool(block2)

        #third block
        block3 = self.encoder_3_1_conv(block2)
        if (self.adapters):
            adapt5 = self.adapter_5_1_batchnorm(block3)
            adapt5 = self.adapter_5_2_conv(adapt5)
            block3 = block3 + adapt5
        block3 = self.encoder_3_2_batchnorm(block3) 
        block3 = self.encoder_3_3_relu(block3)
        block3 = self.encoder_3_4_conv(block3)
        if (self.adapters):
            adapt6 = self.adapter_6_1_batchnorm(block3)
            adapt6 = self.adapter_6_2_conv(adapt6)
            block3 = block3 + adapt6
        block3 = self.encoder_3_5_batchnorm(block3)
        block3 = self.encoder_3_6_relu(block3)
        block3 = self.encoder_3_7_conv(block3)
        if (self.adapters):
            adapt7 = self.adapter_7_1_batchnorm(block3)
            adapt7 = self.adapter_7_2_conv(adapt7)
            block3 = block3 + adapt7
        block3 = self.encoder_3_8_batchnorm(block3)
        block3 = self.encoder_3_9_relu(block3)
        block3 = self.encoder_3_10_conv(block3)
        if (self.adapters):
            adapt8 = self.adapter_8_1_batchnorm(block3)
            adapt8 = self.adapter_8_2_conv(adapt8)
            block3 = block3 + adapt8
        block3 = self.encoder_3_11_batchnorm(block3)
        block3 = self.encoder_3_12_relu(block3)
        x3 = block3 #save it for decoder
        block3 = self.encoder_3_13_maxpool(block3)

        #fourth block
        block4 = self.encoder_4_1_conv(block3)
        if (self.adapters):
            adapt9 = self.adapter_9_1_batchnorm(block4)
            adapt9 = self.adapter_9_2_conv(adapt9)
            block4 = block4 + adapt9
        block4 = self.encoder_4_2_batchnorm(block4) 
        block4 = self.encoder_4_3_relu(block4)
        block4 = self.encoder_4_4_conv(block4)
        if (self.adapters):
            adapt10 = self.adapter_10_1_batchnorm(block4)
            adapt10 = self.adapter_10_2_conv(adapt10)
            block4 = block4 + adapt10
        block4 = self.encoder_4_5_batchnorm(block4)
        block4 = self.encoder_4_6_relu(block4)
        block4 = self.encoder_4_7_conv(block4)
        if (self.adapters):
            adapt11 = self.adapter_11_1_batchnorm(block4)
            adapt11 = self.adapter_11_2_conv(adapt11)
            block4 = block4 + adapt11
        block4 = self.encoder_4_8_batchnorm(block4)
        block4 = self.encoder_4_9_relu(block4)
        block4 = self.encoder_4_10_conv(block4)
        if (self.adapters):
            adapt12 = self.adapter_12_1_batchnorm(block4)
            adapt12 = self.adapter_12_2_conv(adapt12)
            block4 = block4 + adapt12
        block4 = self.encoder_4_11_batchnorm(block4)
        block4 = self.encoder_4_12_relu(block4)
        x4 = block4 #save it for decoder
        block4 = self.encoder_4_13_maxpool(block4)

        #fifth block
        block5 = self.encoder_5_1_conv(block4)
        if (self.adapters):
            adapt13 = self.adapter_13_1_batchnorm(block5)
            adapt13 = self.adapter_13_2_conv(adapt13)
            block5 = block5 + adapt13
        block5 = self.encoder_5_2_batchnorm(block5) 
        block5 = self.encoder_5_3_relu(block5)
        block5 = self.encoder_5_4_conv(block5)
        if (self.adapters):
            adapt14 = self.adapter_14_1_batchnorm(block5)
            adapt14 = self.adapter_14_2_conv(adapt14)
            block5 = block5 + adapt14
        block5 = self.encoder_5_5_batchnorm(block5)
        block5 = self.encoder_5_6_relu(block5)
        block5 = self.encoder_5_7_conv(block5)
        if (self.adapters):
            adapt15 = self.adapter_15_1_batchnorm(block5)
            adapt15 = self.adapter_15_2_conv(adapt15)
            block5 = block5 + adapt15
        block5 = self.encoder_5_8_batchnorm(block5)
        block5 = self.encoder_5_9_relu(block5)
        block5 = self.encoder_5_10_conv(block5)
        if (self.adapters):
            adapt16 = self.adapter_16_1_batchnorm(block5)
            adapt16 = self.adapter_16_2_conv(adapt16)
            block5 = block5 + adapt16
        block5 = self.encoder_5_11_batchnorm(block5)
        block5 = self.encoder_5_12_relu(block5)
        x5 = block5 #bottleneck
        block5 = self.encoder_5_13_maxpool(block5)
        #block5 = self.encoder_5_14_avgpool(block5)

        #center block
        deblock1 = self.decoder_1_1_conv(block5)
        if (self.adapters):
            adapt17 = self.adapter_17_1_batchnorm(deblock1)
            adapt17 = self.adapter_17_2_conv(adapt17)
            deblock1 = deblock1 + adapt17
        deblock1 = self.decoder_1_2_batchnorm(deblock1)
        deblock1 = self.decoder_1_3_relu(deblock1)
        deblock1 = self.decoder_1_4_conv(deblock1)
        if (self.adapters):
            adapt18 = self.adapter_18_1_batchnorm(deblock1)
            adapt18 = self.adapter_18_2_conv(adapt18)
            deblock1 = deblock1 + adapt18
        deblock1 = self.decoder_1_5_batchnorm(deblock1)
        deblock1 = self.decoder_1_6_relu(deblock1)
        x6 = deblock1 #after center block

        # print("Encoder Shapes: ")
        # print(block1.size(),block2.size(),block3.size(),block4.size(),block5.size())

        #upsampling + concatenation
        y1 = F.interpolate(x6, scale_factor=2, mode="nearest")
        y1 = torch.cat([y1,x5],dim=1) #512 + 512

        #first decoder block
        deblock2 = self.decoder_2_1_conv(y1)
        if (self.adapters):
            adapt19 = self.adapter_19_1_batchnorm(deblock2)
            adapt19 = self.adapter_19_2_conv(adapt19)
            deblock2 = deblock2 + adapt19
        deblock2 = self.decoder_2_2_batchnorm(deblock2)
        deblock2 = self.decoder_2_3_relu(deblock2)
        deblock2 = self.decoder_2_4_conv(deblock2)
        if (self.adapters):
            adapt20 = self.adapter_20_1_batchnorm(deblock2)
            adapt20 = self.adapter_20_2_conv(adapt20)
            deblock2 = deblock2 + adapt20
        deblock2 = self.decoder_2_5_batchnorm(deblock2)
        deblock2 = self.decoder_2_6_relu(deblock2)

        #upsampling + concatenation
        deblock2 = F.interpolate(deblock2, scale_factor=2, mode="nearest")
        y2 = torch.cat([deblock2,x4],dim=1) #256 + 512

        #second decoder block
        deblock3 = self.decoder_3_1_conv(y2)
        if (self.adapters):
            adapt21 = self.adapter_21_1_batchnorm(deblock3)
            adapt21 = self.adapter_21_2_conv(adapt21)
            deblock3 = deblock3 + adapt21
        deblock3 = self.decoder_3_2_batchnorm(deblock3)
        deblock3 = self.decoder_3_3_relu(deblock3)
        deblock3 = self.decoder_3_4_conv(deblock3)
        if (self.adapters):
            adapt22 = self.adapter_22_1_batchnorm(deblock3)
            adapt22 = self.adapter_22_2_conv(adapt22)
            deblock3 = deblock3 + adapt22
        deblock3 = self.decoder_3_5_batchnorm(deblock3)
        deblock3 = self.decoder_3_6_relu(deblock3)

        #upsampling + concatenation
        deblock3 = F.interpolate(deblock3, scale_factor=2, mode="nearest")
        y3 = torch.cat([deblock3,x3],dim=1) #128 + 256

        #third decoder block
        deblock4 = self.decoder_4_1_conv(y3)
        if (self.adapters):
            adapt23 = self.adapter_23_1_batchnorm(deblock4)
            adapt23 = self.adapter_23_2_conv(adapt23)
            deblock4 = deblock4 + adapt23
        deblock4 = self.decoder_4_2_batchnorm(deblock4)
        deblock4 = self.decoder_4_3_relu(deblock4)
        deblock4 = self.decoder_4_4_conv(deblock4)
        if (self.adapters):
            adapt24 = self.adapter_24_1_batchnorm(deblock4)
            adapt24 = self.adapter_24_2_conv(adapt24)
            deblock4 = deblock4 + adapt24
        deblock4 = self.decoder_4_5_batchnorm(deblock4)
        deblock4 = self.decoder_4_6_relu(deblock4)

        #upsampling + concatenation
        deblock4 = F.interpolate(deblock4, scale_factor=2, mode="nearest")
        y4 = torch.cat([deblock4,x2],dim=1) #64 + 128

        #fourth decoder block
        deblock5 = self.decoder_5_1_conv(y4)
        if (self.adapters):
            adapt25 = self.adapter_25_1_batchnorm(deblock5)
            adapt25 = self.adapter_25_2_conv(adapt25)
            deblock5 = deblock5 + adapt25
        deblock5 = self.decoder_5_2_batchnorm(deblock5)
        deblock5 = self.decoder_5_3_relu(deblock5)
        deblock5 = self.decoder_5_4_conv(deblock5)
        if (self.adapters):
            adapt26 = self.adapter_26_1_batchnorm(deblock5)
            adapt26 = self.adapter_26_2_conv(adapt26)
            deblock5 = deblock5 + adapt26
        deblock5 = self.decoder_5_5_batchnorm(deblock5)
        deblock5 = self.decoder_5_6_relu(deblock5)

        #no concatenation
        deblock5 = F.interpolate(deblock5, scale_factor=2, mode="nearest")
        y5 = deblock5 #32

        #fifth decoder block (segmentation head)
        deblock6 = self.decoder_6_1_conv(y5)
        if (self.adapters):
            adapt27 = self.adapter_27_1_batchnorm(deblock6)
            adapt27 = self.adapter_27_2_conv(adapt27)
            deblock6 = deblock6 + adapt27
        deblock6 = self.decoder_6_2_batchnorm(deblock6)
        deblock6 = self.decoder_6_3_relu(deblock6)
        deblock6 = self.decoder_6_4_conv(deblock6)
        if (self.adapters):
            adapt28 = self.adapter_28_1_batchnorm(deblock6)
            adapt28 = self.adapter_28_2_conv(adapt28)
            deblock6 = deblock6 + adapt28
        deblock6 = self.decoder_6_5_batchnorm(deblock6)
        deblock6 = self.decoder_6_6_relu(deblock6)

        #no concatenation (no upsampling in seg head)
        y6 = deblock6 #16

        #sixth decoder block
        deblock7 = self.decoder_7_1_conv(y6)

        # print("Decoder Shapes:")
        # print(deblock1.size(),deblock2.size(),deblock3.size(),deblock4.size(),deblock5.size(),deblock6.size(),deblock7.size())

        return deblock7

#vgg19_bn
class Unet_vgg19_ranked(nn.Module):

    def __init__(self, in_channels, num_classes, adapters):

        super(Unet_vgg19_ranked, self).__init__()
        self.adapters = adapters

        #----------------------------------------------------------------        
        #encoder layers (encoder_block_layernumber_typeoflayer)
        #----------------------------------------------------------------

        #first block
        self.encoder_1_1_conv = nn.Conv2d(in_channels, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.encoder_1_2_batchnorm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.encoder_1_3_relu = nn.ReLU(inplace=True)
        self.encoder_1_4_conv = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) 
        self.encoder_1_5_batchnorm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.encoder_1_6_relu = nn.ReLU(inplace=True)
        self.encoder_1_7_maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

        if (self.adapters):
            self.adapter_1_1_batchnorm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_1_2_conv = nn.Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_1_2_conv)
            self.init_bnorm(self.adapter_1_1_batchnorm)
            self.adapter_2_1_batchnorm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_2_2_conv = nn.Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_2_2_conv)
            self.init_bnorm(self.adapter_2_1_batchnorm)

        #second block
        self.encoder_2_1_conv = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.encoder_2_2_batchnorm = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.encoder_2_3_relu = nn.ReLU(inplace=True)
        self.encoder_2_4_conv = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.encoder_2_5_batchnorm = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.encoder_2_6_relu = nn.ReLU(inplace=True)
        self.encoder_2_7_maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

        if (self.adapters):
            self.adapter_3_1_batchnorm = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_3_2_conv = nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_3_2_conv)
            self.init_bnorm(self.adapter_3_1_batchnorm)
            self.adapter_4_1_batchnorm = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_4_2_conv = nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_4_2_conv)
            self.init_bnorm(self.adapter_4_1_batchnorm)
      
        #third block
        self.encoder_3_1_conv = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.encoder_3_2_batchnorm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.encoder_3_3_relu = nn.ReLU(inplace=True)
        self.encoder_3_4_conv = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.encoder_3_5_batchnorm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.encoder_3_6_relu = nn.ReLU(inplace=True)
        self.encoder_3_7_conv = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.encoder_3_8_batchnorm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.encoder_3_9_relu = nn.ReLU(inplace=True)
        self.encoder_3_10_conv = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.encoder_3_11_batchnorm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.encoder_3_12_relu = nn.ReLU(inplace=True)
        self.encoder_3_13_maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

        if (self.adapters):
            self.adapter_5_1_batchnorm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_5_2_conv = nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_5_2_conv)
            self.init_bnorm(self.adapter_5_1_batchnorm)
            self.adapter_6_1_batchnorm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_6_2_conv = nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_6_2_conv)
            self.init_bnorm(self.adapter_6_1_batchnorm)
            self.adapter_7_1_batchnorm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_7_2_conv = nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_7_2_conv)
            self.init_bnorm(self.adapter_7_1_batchnorm)
            self.adapter_8_1_batchnorm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_8_2_conv = nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_8_2_conv)
            self.init_bnorm(self.adapter_8_1_batchnorm)
      
        #fourth block
        self.encoder_4_1_conv = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.encoder_4_2_batchnorm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.encoder_4_3_relu = nn.ReLU(inplace=True)
        self.encoder_4_4_conv = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.encoder_4_5_batchnorm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.encoder_4_6_relu = nn.ReLU(inplace=True)
        self.encoder_4_7_conv = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.encoder_4_8_batchnorm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.encoder_4_9_relu = nn.ReLU(inplace=True)
        self.encoder_4_10_conv = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.encoder_4_11_batchnorm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.encoder_4_12_relu = nn.ReLU(inplace=True)
        self.encoder_4_13_maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

        if (self.adapters):
            self.adapter_9_1_batchnorm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_9_2_conv = nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_9_2_conv)
            self.init_bnorm(self.adapter_9_1_batchnorm)
            self.adapter_10_1_batchnorm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_10_2_conv = nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_10_2_conv)
            self.init_bnorm(self.adapter_10_1_batchnorm)
            self.adapter_11_1_batchnorm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_11_2_conv = nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_11_2_conv)
            self.init_bnorm(self.adapter_11_1_batchnorm)
            self.adapter_12_1_batchnorm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_12_2_conv = nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_12_2_conv)
            self.init_bnorm(self.adapter_12_1_batchnorm)
    
        #fifth block
        self.encoder_5_1_conv = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.encoder_5_2_batchnorm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.encoder_5_3_relu = nn.ReLU(inplace=True)
        self.encoder_5_4_conv = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.encoder_5_5_batchnorm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.encoder_5_6_relu = nn.ReLU(inplace=True)
        self.encoder_5_7_conv = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.encoder_5_8_batchnorm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.encoder_5_9_relu = nn.ReLU(inplace=True)
        self.encoder_5_10_conv = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.encoder_5_11_batchnorm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.encoder_5_12_relu = nn.ReLU(inplace=True)
        self.encoder_5_13_maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.encoder_5_14_avgpool = nn.AdaptiveAvgPool2d(output_size=(7, 7))

        if (self.adapters):
            self.adapter_13_1_batchnorm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_13_2_conv = nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_13_2_conv)
            self.init_bnorm(self.adapter_13_1_batchnorm)
            self.adapter_14_1_batchnorm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_14_2_conv = nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_14_2_conv)
            self.init_bnorm(self.adapter_14_1_batchnorm)
            self.adapter_15_1_batchnorm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_15_2_conv = nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_15_2_conv)
            self.init_bnorm(self.adapter_15_1_batchnorm)
            self.adapter_16_1_batchnorm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_16_2_conv = nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_16_2_conv)
            self.init_bnorm(self.adapter_16_1_batchnorm)

        #----------------------------------------------------------------        
        #decoder layers (decoder_block_layernumber_typeoflayer)
        #----------------------------------------------------------------

        #center block
        self.decoder_1_1_conv = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.decoder_1_2_batchnorm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.decoder_1_3_relu = nn.ReLU(inplace=True)
        self.decoder_1_4_conv = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.decoder_1_5_batchnorm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.decoder_1_6_relu = nn.ReLU(inplace=True)

        if (self.adapters):
            self.adapter_17_1_batchnorm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_17_2_conv = nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_17_2_conv)
            self.init_bnorm(self.adapter_17_1_batchnorm)
            self.adapter_18_1_batchnorm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_18_2_conv = nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_18_2_conv)
            self.init_bnorm(self.adapter_18_1_batchnorm)

        #first decoder block
        self.decoder_2_1_conv = nn.Conv2d(1024, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.decoder_2_2_batchnorm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.decoder_2_3_relu = nn.ReLU(inplace=True)
        self.decoder_2_4_conv = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.decoder_2_5_batchnorm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.decoder_2_6_relu = nn.ReLU(inplace=True)

        if (self.adapters):
            self.adapter_19_1_batchnorm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_19_2_conv = nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_19_2_conv)
            self.init_bnorm(self.adapter_19_1_batchnorm)
            self.adapter_20_1_batchnorm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_20_2_conv = nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_20_2_conv)
            self.init_bnorm(self.adapter_20_1_batchnorm)

        #second decoder block
        self.decoder_3_1_conv = nn.Conv2d(768, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.decoder_3_2_batchnorm = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.decoder_3_3_relu = nn.ReLU(inplace=True)
        self.decoder_3_4_conv = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.decoder_3_5_batchnorm = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.decoder_3_6_relu = nn.ReLU(inplace=True)

        if (self.adapters):
            self.adapter_21_1_batchnorm = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_21_2_conv = nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_21_2_conv)
            self.init_bnorm(self.adapter_21_1_batchnorm)
            self.adapter_22_1_batchnorm = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_22_2_conv = nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_22_2_conv)
            self.init_bnorm(self.adapter_22_1_batchnorm)

        #third decoder block
        self.decoder_4_1_conv = nn.Conv2d(384, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.decoder_4_2_batchnorm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.decoder_4_3_relu = nn.ReLU(inplace=True)
        self.decoder_4_4_conv = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.decoder_4_5_batchnorm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.decoder_4_6_relu = nn.ReLU(inplace=True)

        if (self.adapters):
            self.adapter_23_1_batchnorm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_23_2_conv = nn.Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_23_2_conv)
            self.init_bnorm(self.adapter_23_1_batchnorm)
            self.adapter_24_1_batchnorm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_24_2_conv = nn.Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_24_2_conv)
            self.init_bnorm(self.adapter_24_1_batchnorm)

        #fourth decoder block
        self.decoder_5_1_conv = nn.Conv2d(192, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.decoder_5_2_batchnorm = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.decoder_5_3_relu = nn.ReLU(inplace=True)
        self.decoder_5_4_conv = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.decoder_5_5_batchnorm = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.decoder_5_6_relu = nn.ReLU(inplace=True)

        if (self.adapters):
            self.adapter_25_1_batchnorm = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_25_2_conv = nn.Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_25_2_conv)
            self.init_bnorm(self.adapter_25_1_batchnorm)
            self.adapter_26_1_batchnorm = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_26_2_conv = nn.Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_26_2_conv)
            self.init_bnorm(self.adapter_26_1_batchnorm)

        #fifth decoder block
        self.decoder_6_1_conv = nn.Conv2d(32, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.decoder_6_2_batchnorm = nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.decoder_6_3_relu = nn.ReLU(inplace=True)
        self.decoder_6_4_conv = nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.decoder_6_5_batchnorm = nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.decoder_6_6_relu = nn.ReLU(inplace=True)

        if (self.adapters):
            self.adapter_27_1_batchnorm = nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_27_2_conv = nn.Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_27_2_conv)
            self.init_bnorm(self.adapter_27_1_batchnorm)
            self.adapter_28_1_batchnorm = nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            self.adapter_28_2_conv = nn.Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.init_convweights(self.adapter_28_2_conv)
            self.init_bnorm(self.adapter_28_1_batchnorm)

        #final segmentation head
        self.decoder_7_1_conv = nn.Conv2d(16, num_classes, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    def init_convweights(self, layer):
        layer.weight.data.zero_()
        return

    def init_bnorm(self, layer):
        layer.weight.data.fill_(1)
        layer.bias.data.zero_()
        layer.running_mean.zero_()
        layer.running_var.fill_(1)
        layer.num_batches_tracked.zero_()
        return
    
    def forward(self, x):

        # print("Input Shape:")
        # print(x.size())

        # if (self.adapters):
        #     adapt1 = self.adapter_1_1_batchnorm(block1)
        #     adapt1 = self.adapter_1_2_conv(adapt1)
        #     block1 = block1 + adapt1

        #first block
        block1 = self.encoder_1_1_conv(x)
        if (self.adapters):
            adapt1 = self.adapter_1_1_batchnorm(block1)
            adapt1 = self.adapter_1_2_conv(adapt1)
            block1 = block1 + adapt1
        block1 = self.encoder_1_2_batchnorm(block1)
        block1 = self.encoder_1_3_relu(block1)
        block1 = self.encoder_1_4_conv(block1)
        if (self.adapters):
            adapt2 = self.adapter_2_1_batchnorm(block1)
            adapt2 = self.adapter_2_2_conv(adapt2)
            block1 = block1 + adapt2
        block1 = self.encoder_1_5_batchnorm(block1)
        block1 = self.encoder_1_6_relu(block1)
        x1 = block1 #save it for decoder
        block1 = self.encoder_1_7_maxpool(block1)

        #second block
        block2 = self.encoder_2_1_conv(block1)
        if (self.adapters):
            adapt3 = self.adapter_3_1_batchnorm(block2)
            adapt3 = self.adapter_3_2_conv(adapt3)
            block2 = block2 + adapt3
        block2 = self.encoder_2_2_batchnorm(block2) 
        block2 = self.encoder_2_3_relu(block2)
        block2 = self.encoder_2_4_conv(block2)
        if (self.adapters):
            adapt4 = self.adapter_4_1_batchnorm(block2)
            adapt4 = self.adapter_4_2_conv(adapt4)
            block2 = block2 + adapt4
        block2 = self.encoder_2_5_batchnorm(block2)
        block2 = self.encoder_2_6_relu(block2)
        x2 = block2 #save it for decoder
        block2 = self.encoder_2_7_maxpool(block2)

        #third block
        block3 = self.encoder_3_1_conv(block2)
        if (self.adapters):
            adapt5 = self.adapter_5_1_batchnorm(block3)
            adapt5 = self.adapter_5_2_conv(adapt5)
            block3 = block3 + adapt5
        block3 = self.encoder_3_2_batchnorm(block3) 
        block3 = self.encoder_3_3_relu(block3)
        block3 = self.encoder_3_4_conv(block3)
        if (self.adapters):
            adapt6 = self.adapter_6_1_batchnorm(block3)
            adapt6 = self.adapter_6_2_conv(adapt6)
            block3 = block3 + adapt6
        block3 = self.encoder_3_5_batchnorm(block3)
        block3 = self.encoder_3_6_relu(block3)
        block3 = self.encoder_3_7_conv(block3)
        if (self.adapters):
            adapt7 = self.adapter_7_1_batchnorm(block3)
            adapt7 = self.adapter_7_2_conv(adapt7)
            block3 = block3 + adapt7
        block3 = self.encoder_3_8_batchnorm(block3)
        block3 = self.encoder_3_9_relu(block3)
        block3 = self.encoder_3_10_conv(block3)
        if (self.adapters):
            adapt8 = self.adapter_8_1_batchnorm(block3)
            adapt8 = self.adapter_8_2_conv(adapt8)
            block3 = block3 + adapt8
        block3 = self.encoder_3_11_batchnorm(block3)
        block3 = self.encoder_3_12_relu(block3)
        x3 = block3 #save it for decoder
        block3 = self.encoder_3_13_maxpool(block3)

        #fourth block
        block4 = self.encoder_4_1_conv(block3)
        if (self.adapters):
            adapt9 = self.adapter_9_1_batchnorm(block4)
            adapt9 = self.adapter_9_2_conv(adapt9)
            block4 = block4 + adapt9
        block4 = self.encoder_4_2_batchnorm(block4) 
        block4 = self.encoder_4_3_relu(block4)
        block4 = self.encoder_4_4_conv(block4)
        block4 = self.encoder_4_5_batchnorm(block4)
        block4 = self.encoder_4_6_relu(block4)
        block4 = self.encoder_4_7_conv(block4)
        block4 = self.encoder_4_8_batchnorm(block4)
        block4 = self.encoder_4_9_relu(block4)
        block4 = self.encoder_4_10_conv(block4)
        block4 = self.encoder_4_11_batchnorm(block4)
        block4 = self.encoder_4_12_relu(block4)
        x4 = block4 #save it for decoder
        block4 = self.encoder_4_13_maxpool(block4)

        #fifth block
        block5 = self.encoder_5_1_conv(block4)
        block5 = self.encoder_5_2_batchnorm(block5) 
        block5 = self.encoder_5_3_relu(block5)
        block5 = self.encoder_5_4_conv(block5)
        block5 = self.encoder_5_5_batchnorm(block5)
        block5 = self.encoder_5_6_relu(block5)
        block5 = self.encoder_5_7_conv(block5)
        block5 = self.encoder_5_8_batchnorm(block5)
        block5 = self.encoder_5_9_relu(block5)
        block5 = self.encoder_5_10_conv(block5)
        block5 = self.encoder_5_11_batchnorm(block5)
        block5 = self.encoder_5_12_relu(block5)
        x5 = block5 #bottleneck
        block5 = self.encoder_5_13_maxpool(block5)
        #block5 = self.encoder_5_14_avgpool(block5)

        #center block
        deblock1 = self.decoder_1_1_conv(block5)
        deblock1 = self.decoder_1_2_batchnorm(deblock1)
        deblock1 = self.decoder_1_3_relu(deblock1)
        deblock1 = self.decoder_1_4_conv(deblock1)
        deblock1 = self.decoder_1_5_batchnorm(deblock1)
        deblock1 = self.decoder_1_6_relu(deblock1)
        x6 = deblock1 #after center block

        # print("Encoder Shapes: ")
        # print(block1.size(),block2.size(),block3.size(),block4.size(),block5.size())

        #upsampling + concatenation
        y1 = F.interpolate(x6, scale_factor=2, mode="nearest")
        y1 = torch.cat([y1,x5],dim=1) #512 + 512

        #first decoder block
        deblock2 = self.decoder_2_1_conv(y1)
        if (self.adapters):
            adapt19 = self.adapter_19_1_batchnorm(deblock2)
            adapt19 = self.adapter_19_2_conv(adapt19)
            deblock2 = deblock2 + adapt19
        deblock2 = self.decoder_2_2_batchnorm(deblock2)
        deblock2 = self.decoder_2_3_relu(deblock2)
        deblock2 = self.decoder_2_4_conv(deblock2)
        if (self.adapters):
            adapt20 = self.adapter_20_1_batchnorm(deblock2)
            adapt20 = self.adapter_20_2_conv(adapt20)
            deblock2 = deblock2 + adapt20
        deblock2 = self.decoder_2_5_batchnorm(deblock2)
        deblock2 = self.decoder_2_6_relu(deblock2)

        #upsampling + concatenation
        deblock2 = F.interpolate(deblock2, scale_factor=2, mode="nearest")
        y2 = torch.cat([deblock2,x4],dim=1) #256 + 512

        #second decoder block
        deblock3 = self.decoder_3_1_conv(y2)
        if (self.adapters):
            adapt21 = self.adapter_21_1_batchnorm(deblock3)
            adapt21 = self.adapter_21_2_conv(adapt21)
            deblock3 = deblock3 + adapt21
        deblock3 = self.decoder_3_2_batchnorm(deblock3)
        deblock3 = self.decoder_3_3_relu(deblock3)
        deblock3 = self.decoder_3_4_conv(deblock3)
        if (self.adapters):
            adapt22 = self.adapter_22_1_batchnorm(deblock3)
            adapt22 = self.adapter_22_2_conv(adapt22)
            deblock3 = deblock3 + adapt22
        deblock3 = self.decoder_3_5_batchnorm(deblock3)
        deblock3 = self.decoder_3_6_relu(deblock3)

        #upsampling + concatenation
        deblock3 = F.interpolate(deblock3, scale_factor=2, mode="nearest")
        y3 = torch.cat([deblock3,x3],dim=1) #128 + 256

        #third decoder block
        deblock4 = self.decoder_4_1_conv(y3)
        if (self.adapters):
            adapt23 = self.adapter_23_1_batchnorm(deblock4)
            adapt23 = self.adapter_23_2_conv(adapt23)
            deblock4 = deblock4 + adapt23
        deblock4 = self.decoder_4_2_batchnorm(deblock4)
        deblock4 = self.decoder_4_3_relu(deblock4)
        deblock4 = self.decoder_4_4_conv(deblock4)
        if (self.adapters):
            adapt24 = self.adapter_24_1_batchnorm(deblock4)
            adapt24 = self.adapter_24_2_conv(adapt24)
            deblock4 = deblock4 + adapt24
        deblock4 = self.decoder_4_5_batchnorm(deblock4)
        deblock4 = self.decoder_4_6_relu(deblock4)

        #upsampling + concatenation
        deblock4 = F.interpolate(deblock4, scale_factor=2, mode="nearest")
        y4 = torch.cat([deblock4,x2],dim=1) #64 + 128

        #fourth decoder block
        deblock5 = self.decoder_5_1_conv(y4)
        if (self.adapters):
            adapt25 = self.adapter_25_1_batchnorm(deblock5)
            adapt25 = self.adapter_25_2_conv(adapt25)
            deblock5 = deblock5 + adapt25
        deblock5 = self.decoder_5_2_batchnorm(deblock5)
        deblock5 = self.decoder_5_3_relu(deblock5)
        deblock5 = self.decoder_5_4_conv(deblock5)
        if (self.adapters):
            adapt26 = self.adapter_26_1_batchnorm(deblock5)
            adapt26 = self.adapter_26_2_conv(adapt26)
            deblock5 = deblock5 + adapt26
        deblock5 = self.decoder_5_5_batchnorm(deblock5)
        deblock5 = self.decoder_5_6_relu(deblock5)

        #no concatenation
        deblock5 = F.interpolate(deblock5, scale_factor=2, mode="nearest")
        y5 = deblock5 #32

        #fifth decoder block (segmentation head)
        deblock6 = self.decoder_6_1_conv(y5)
        if (self.adapters):
            adapt27 = self.adapter_27_1_batchnorm(deblock6)
            adapt27 = self.adapter_27_2_conv(adapt27)
            deblock6 = deblock6 + adapt27
        deblock6 = self.decoder_6_2_batchnorm(deblock6)
        deblock6 = self.decoder_6_3_relu(deblock6)
        deblock6 = self.decoder_6_4_conv(deblock6)
        if (self.adapters):
            adapt28 = self.adapter_28_1_batchnorm(deblock6)
            adapt28 = self.adapter_28_2_conv(adapt28)
            deblock6 = deblock6 + adapt28
        deblock6 = self.decoder_6_5_batchnorm(deblock6)
        deblock6 = self.decoder_6_6_relu(deblock6)

        #no concatenation (no upsampling in seg head)
        y6 = deblock6 #16

        #sixth decoder block
        deblock7 = self.decoder_7_1_conv(y6)

        # print("Decoder Shapes:")
        # print(deblock1.size(),deblock2.size(),deblock3.size(),deblock4.size(),deblock5.size(),deblock6.size(),deblock7.size())

        return deblock7

#vgg19_bn
class Unet_vgg19_fuse(nn.Module):

    def __init__(self, in_channels, num_classes):

        super(Unet_vgg19_fuse, self).__init__()

        #----------------------------------------------------------------        
        #encoder layers (encoder_block_layernumber_typeoflayer)
        #----------------------------------------------------------------

        #first block
        self.encoder_1_1_conv = nn.Conv2d(in_channels, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.encoder_1_3_relu = nn.ReLU(inplace=True)
        self.encoder_1_4_conv = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) 
        self.encoder_1_6_relu = nn.ReLU(inplace=True)
        self.encoder_1_7_maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

        #second block
        self.encoder_2_1_conv = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.encoder_2_3_relu = nn.ReLU(inplace=True)
        self.encoder_2_4_conv = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.encoder_2_6_relu = nn.ReLU(inplace=True)
        self.encoder_2_7_maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      
        #third block
        self.encoder_3_1_conv = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.encoder_3_3_relu = nn.ReLU(inplace=True)
        self.encoder_3_4_conv = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.encoder_3_6_relu = nn.ReLU(inplace=True)
        self.encoder_3_7_conv = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.encoder_3_9_relu = nn.ReLU(inplace=True)
        self.encoder_3_10_conv = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.encoder_3_12_relu = nn.ReLU(inplace=True)
        self.encoder_3_13_maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      
        #fourth block
        self.encoder_4_1_conv = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.encoder_4_3_relu = nn.ReLU(inplace=True)
        self.encoder_4_4_conv = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.encoder_4_6_relu = nn.ReLU(inplace=True)
        self.encoder_4_7_conv = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.encoder_4_9_relu = nn.ReLU(inplace=True)
        self.encoder_4_10_conv = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.encoder_4_12_relu = nn.ReLU(inplace=True)
        self.encoder_4_13_maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    
        #fifth block
        self.encoder_5_1_conv = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.encoder_5_3_relu = nn.ReLU(inplace=True)
        self.encoder_5_4_conv = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.encoder_5_6_relu = nn.ReLU(inplace=True)
        self.encoder_5_7_conv = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.encoder_5_9_relu = nn.ReLU(inplace=True)
        self.encoder_5_10_conv = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.encoder_5_12_relu = nn.ReLU(inplace=True)
        self.encoder_5_13_maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.encoder_5_14_avgpool = nn.AdaptiveAvgPool2d(output_size=(7, 7))

        #----------------------------------------------------------------        
        #decoder layers (decoder_block_layernumber_typeoflayer)
        #----------------------------------------------------------------

        #center block
        self.decoder_1_1_conv = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.decoder_1_3_relu = nn.ReLU(inplace=True)
        self.decoder_1_4_conv = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.decoder_1_6_relu = nn.ReLU(inplace=True)

        #first decoder block
        self.decoder_2_1_conv = nn.Conv2d(1024, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.decoder_2_3_relu = nn.ReLU(inplace=True)
        self.decoder_2_4_conv = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.decoder_2_6_relu = nn.ReLU(inplace=True)

        #second decoder block
        self.decoder_3_1_conv = nn.Conv2d(768, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.decoder_3_3_relu = nn.ReLU(inplace=True)
        self.decoder_3_4_conv = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.decoder_3_6_relu = nn.ReLU(inplace=True)

        #third decoder block
        self.decoder_4_1_conv = nn.Conv2d(384, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.decoder_4_3_relu = nn.ReLU(inplace=True)
        self.decoder_4_4_conv = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.decoder_4_6_relu = nn.ReLU(inplace=True)

        #fourth decoder block
        self.decoder_5_1_conv = nn.Conv2d(192, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.decoder_5_3_relu = nn.ReLU(inplace=True)
        self.decoder_5_4_conv = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.decoder_5_6_relu = nn.ReLU(inplace=True)

        #fifth decoder block
        self.decoder_6_1_conv = nn.Conv2d(32, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.decoder_6_3_relu = nn.ReLU(inplace=True)
        self.decoder_6_4_conv = nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.decoder_6_6_relu = nn.ReLU(inplace=True)

        #final segmentation head
        self.decoder_7_1_conv = nn.Conv2d(16, num_classes, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    def forward(self, x):

        #first block
        block1 = self.encoder_1_1_conv(x)
        block1 = self.encoder_1_3_relu(block1)
        block1 = self.encoder_1_4_conv(block1)
        block1 = self.encoder_1_6_relu(block1)
        x1 = block1 #save it for decoder
        block1 = self.encoder_1_7_maxpool(block1)

        #second block
        block2 = self.encoder_2_1_conv(block1)
        block2 = self.encoder_2_3_relu(block2)
        block2 = self.encoder_2_4_conv(block2)
        block2 = self.encoder_2_6_relu(block2)
        x2 = block2 #save it for decoder
        block2 = self.encoder_2_7_maxpool(block2)

        #third block
        block3 = self.encoder_3_1_conv(block2)
        block3 = self.encoder_3_3_relu(block3)
        block3 = self.encoder_3_4_conv(block3)
        block3 = self.encoder_3_6_relu(block3)
        block3 = self.encoder_3_7_conv(block3)
        block3 = self.encoder_3_9_relu(block3)
        block3 = self.encoder_3_10_conv(block3)
        block3 = self.encoder_3_12_relu(block3)
        x3 = block3 #save it for decoder
        block3 = self.encoder_3_13_maxpool(block3)

        #fourth block
        block4 = self.encoder_4_1_conv(block3)
        block4 = self.encoder_4_3_relu(block4)
        block4 = self.encoder_4_4_conv(block4)
        block4 = self.encoder_4_6_relu(block4)
        block4 = self.encoder_4_7_conv(block4)
        block4 = self.encoder_4_9_relu(block4)
        block4 = self.encoder_4_10_conv(block4)
        block4 = self.encoder_4_12_relu(block4)
        x4 = block4 #save it for decoder
        block4 = self.encoder_4_13_maxpool(block4)

        #fifth block
        block5 = self.encoder_5_1_conv(block4)
        block5 = self.encoder_5_3_relu(block5)
        block5 = self.encoder_5_4_conv(block5)
        block5 = self.encoder_5_6_relu(block5)
        block5 = self.encoder_5_7_conv(block5)
        block5 = self.encoder_5_9_relu(block5)
        block5 = self.encoder_5_10_conv(block5)
        block5 = self.encoder_5_12_relu(block5)
        x5 = block5 #bottleneck
        block5 = self.encoder_5_13_maxpool(block5)

        #center block
        deblock1 = self.decoder_1_1_conv(block5)
        deblock1 = self.decoder_1_3_relu(deblock1)
        deblock1 = self.decoder_1_4_conv(deblock1)
        deblock1 = self.decoder_1_6_relu(deblock1)
        x6 = deblock1 #after center block

        #upsampling + concatenation
        y1 = F.interpolate(x6, scale_factor=2, mode="nearest")
        y1 = torch.cat([y1,x5],dim=1) #512 + 512

        #first decoder block
        deblock2 = self.decoder_2_1_conv(y1)
        deblock2 = self.decoder_2_3_relu(deblock2)
        deblock2 = self.decoder_2_4_conv(deblock2)
        deblock2 = self.decoder_2_6_relu(deblock2)

        #upsampling + concatenation
        deblock2 = F.interpolate(deblock2, scale_factor=2, mode="nearest")
        y2 = torch.cat([deblock2,x4],dim=1) #256 + 512

        #second decoder block
        deblock3 = self.decoder_3_1_conv(y2)
        deblock3 = self.decoder_3_3_relu(deblock3)
        deblock3 = self.decoder_3_4_conv(deblock3)
        deblock3 = self.decoder_3_6_relu(deblock3)

        #upsampling + concatenation
        deblock3 = F.interpolate(deblock3, scale_factor=2, mode="nearest")
        y3 = torch.cat([deblock3,x3],dim=1) #128 + 256

        #third decoder block
        deblock4 = self.decoder_4_1_conv(y3)
        deblock4 = self.decoder_4_3_relu(deblock4)
        deblock4 = self.decoder_4_4_conv(deblock4)
        deblock4 = self.decoder_4_6_relu(deblock4)

        #upsampling + concatenation
        deblock4 = F.interpolate(deblock4, scale_factor=2, mode="nearest")
        y4 = torch.cat([deblock4,x2],dim=1) #64 + 128

        #fourth decoder block
        deblock5 = self.decoder_5_1_conv(y4)
        deblock5 = self.decoder_5_3_relu(deblock5)
        deblock5 = self.decoder_5_4_conv(deblock5)
        deblock5 = self.decoder_5_6_relu(deblock5)

        #no concatenation
        deblock5 = F.interpolate(deblock5, scale_factor=2, mode="nearest")
        y5 = deblock5 #32

        #fifth decoder block (segmentation head)
        deblock6 = self.decoder_6_1_conv(y5)
        deblock6 = self.decoder_6_3_relu(deblock6)
        deblock6 = self.decoder_6_4_conv(deblock6)
        deblock6 = self.decoder_6_6_relu(deblock6)

        #no concatenation (no upsampling in seg head)
        y6 = deblock6 #16

        #sixth decoder block
        deblock7 = self.decoder_7_1_conv(y6)

        return deblock7

def transfer_weights_vgg19bn(model, model_smp):

    enc_childrens = list(model_smp.encoder.children())
    enc_childrens = list(enc_childrens[0].children())

    #first block
    model.encoder_1_1_conv.load_state_dict(enc_childrens[0].state_dict())
    model.encoder_1_2_batchnorm.load_state_dict(enc_childrens[1].state_dict())
    model.encoder_1_4_conv.load_state_dict(enc_childrens[3].state_dict())
    model.encoder_1_5_batchnorm.load_state_dict(enc_childrens[4].state_dict())

    #second block
    model.encoder_2_1_conv.load_state_dict(enc_childrens[7].state_dict())
    model.encoder_2_2_batchnorm.load_state_dict(enc_childrens[8].state_dict())
    model.encoder_2_4_conv.load_state_dict(enc_childrens[10].state_dict())
    model.encoder_2_5_batchnorm.load_state_dict(enc_childrens[11].state_dict())
    
    #third block
    model.encoder_3_1_conv.load_state_dict(enc_childrens[14].state_dict())
    model.encoder_3_2_batchnorm.load_state_dict(enc_childrens[15].state_dict())
    model.encoder_3_4_conv.load_state_dict(enc_childrens[17].state_dict())
    model.encoder_3_5_batchnorm.load_state_dict(enc_childrens[18].state_dict())
    model.encoder_3_7_conv.load_state_dict(enc_childrens[20].state_dict())
    model.encoder_3_8_batchnorm.load_state_dict(enc_childrens[21].state_dict())
    model.encoder_3_10_conv.load_state_dict(enc_childrens[23].state_dict())
    model.encoder_3_11_batchnorm.load_state_dict(enc_childrens[24].state_dict())
    
    #fourth block
    model.encoder_4_1_conv.load_state_dict(enc_childrens[27].state_dict())
    model.encoder_4_2_batchnorm.load_state_dict(enc_childrens[28].state_dict())
    model.encoder_4_4_conv.load_state_dict(enc_childrens[30].state_dict())
    model.encoder_4_5_batchnorm.load_state_dict(enc_childrens[31].state_dict())
    model.encoder_4_7_conv.load_state_dict(enc_childrens[33].state_dict())
    model.encoder_4_8_batchnorm.load_state_dict(enc_childrens[34].state_dict())
    model.encoder_4_10_conv.load_state_dict(enc_childrens[36].state_dict())
    model.encoder_4_11_batchnorm.load_state_dict(enc_childrens[37].state_dict())

    #fifth block
    model.encoder_5_1_conv.load_state_dict(enc_childrens[40].state_dict())
    model.encoder_5_2_batchnorm.load_state_dict(enc_childrens[41].state_dict())
    model.encoder_5_4_conv.load_state_dict(enc_childrens[43].state_dict())
    model.encoder_5_5_batchnorm.load_state_dict(enc_childrens[44].state_dict())
    model.encoder_5_7_conv.load_state_dict(enc_childrens[46].state_dict())
    model.encoder_5_8_batchnorm.load_state_dict(enc_childrens[47].state_dict())
    model.encoder_5_10_conv.load_state_dict(enc_childrens[49].state_dict())
    model.encoder_5_11_batchnorm.load_state_dict(enc_childrens[50].state_dict())

    #center block
    dec_childrens = list(list(model_smp.decoder.children())[0].children())
    dec_block0_childrens = list(dec_childrens[0].children()) + list(dec_childrens[1].children())
    model.decoder_1_1_conv.load_state_dict(dec_block0_childrens[0].state_dict())
    model.decoder_1_2_batchnorm.load_state_dict(dec_block0_childrens[1].state_dict())
    model.decoder_1_4_conv.load_state_dict(dec_block0_childrens[3].state_dict())
    model.decoder_1_5_batchnorm.load_state_dict(dec_block0_childrens[4].state_dict())

    dec_childrens = list(list(model_smp.decoder.children())[1].children())
    dec_block1_childrens = list(list(dec_childrens[0].children())[0].children()) + list(list(dec_childrens[0].children())[2].children())
    model.decoder_2_1_conv.load_state_dict(dec_block1_childrens[0].state_dict())
    model.decoder_2_2_batchnorm.load_state_dict(dec_block1_childrens[1].state_dict())
    model.decoder_2_4_conv.load_state_dict(dec_block1_childrens[3].state_dict())
    model.decoder_2_5_batchnorm.load_state_dict(dec_block1_childrens[4].state_dict())

    dec_block2_childrens = list(list(dec_childrens[1].children())[0].children()) + list(list(dec_childrens[1].children())[2].children())
    model.decoder_3_1_conv.load_state_dict(dec_block2_childrens[0].state_dict())
    model.decoder_3_2_batchnorm.load_state_dict(dec_block2_childrens[1].state_dict())
    model.decoder_3_4_conv.load_state_dict(dec_block2_childrens[3].state_dict())
    model.decoder_3_5_batchnorm.load_state_dict(dec_block2_childrens[4].state_dict())

    dec_block3_childrens = list(list(dec_childrens[2].children())[0].children()) + list(list(dec_childrens[2].children())[2].children())
    model.decoder_4_1_conv.load_state_dict(dec_block3_childrens[0].state_dict())
    model.decoder_4_2_batchnorm.load_state_dict(dec_block3_childrens[1].state_dict())
    model.decoder_4_4_conv.load_state_dict(dec_block3_childrens[3].state_dict())
    model.decoder_4_5_batchnorm.load_state_dict(dec_block3_childrens[4].state_dict())

    dec_block4_childrens = list(list(dec_childrens[3].children())[0].children()) + list(list(dec_childrens[3].children())[2].children())
    model.decoder_5_1_conv.load_state_dict(dec_block4_childrens[0].state_dict())
    model.decoder_5_2_batchnorm.load_state_dict(dec_block4_childrens[1].state_dict())
    model.decoder_5_4_conv.load_state_dict(dec_block4_childrens[3].state_dict())
    model.decoder_5_5_batchnorm.load_state_dict(dec_block4_childrens[4].state_dict())

    dec_block5_childrens = list(list(dec_childrens[4].children())[0].children()) + list(list(dec_childrens[4].children())[2].children())
    model.decoder_6_1_conv.load_state_dict(dec_block5_childrens[0].state_dict())
    model.decoder_6_2_batchnorm.load_state_dict(dec_block5_childrens[1].state_dict())
    model.decoder_6_4_conv.load_state_dict(dec_block5_childrens[3].state_dict())
    model.decoder_6_5_batchnorm.load_state_dict(dec_block5_childrens[4].state_dict())

    model.decoder_7_1_conv.load_state_dict(list(model_smp.segmentation_head.children())[0].state_dict())

def transfer_weights_resnet18(model, model_smp):

    enc_childrens = list(model_smp.encoder.children())
    model.encoder_1_1_conv.load_state_dict(enc_childrens[0].state_dict())
    model.encoder_1_2_batchnorm.load_state_dict(enc_childrens[1].state_dict())
    model.encoder_1_4_maxpool.load_state_dict(enc_childrens[3].state_dict())

    enc_block2_childrens = list(list(enc_childrens[4].children())[0].children()) + list(list(enc_childrens[4].children())[1].children())
    model.encoder_2_1_conv.load_state_dict(enc_block2_childrens[0].state_dict())
    model.encoder_2_2_batchnorm.load_state_dict(enc_block2_childrens[1].state_dict()) 
    model.encoder_2_4_conv.load_state_dict(enc_block2_childrens[3].state_dict()) 
    model.encoder_2_5_batchnorm.load_state_dict(enc_block2_childrens[4].state_dict())
    model.encoder_2_7_conv.load_state_dict(enc_block2_childrens[5].state_dict())
    model.encoder_2_8_batchnorm.load_state_dict(enc_block2_childrens[6].state_dict())
    model.encoder_2_10_conv.load_state_dict(enc_block2_childrens[8].state_dict())
    model.encoder_2_11_batchnorm.load_state_dict(enc_block2_childrens[9].state_dict())
                                        
    enc_block3_childrens = list(list(enc_childrens[5].children())[0].children())[:-1] + list(list(list(enc_childrens[5].children())[0].children())[-1].children()) + list(list(enc_childrens[5].children())[1].children())
    model.encoder_3_1_conv.load_state_dict(enc_block3_childrens[0].state_dict())
    model.encoder_3_2_batchnorm.load_state_dict(enc_block3_childrens[1].state_dict())
    model.encoder_3_4_conv.load_state_dict(enc_block3_childrens[3].state_dict())
    model.encoder_3_5_batchnorm.load_state_dict(enc_block3_childrens[4].state_dict())
    model.encoder_3_6_downsample.load_state_dict(enc_block3_childrens[5].state_dict())
    model.encoder_3_7_batchnorm.load_state_dict(enc_block3_childrens[6].state_dict())
    model.encoder_3_9_conv.load_state_dict(enc_block3_childrens[7].state_dict())
    model.encoder_3_10_batchnorm.load_state_dict(enc_block3_childrens[8].state_dict())
    model.encoder_3_12_conv.load_state_dict(enc_block3_childrens[10].state_dict())
    model.encoder_3_13_batchnorm.load_state_dict(enc_block3_childrens[11].state_dict())

    enc_block4_childrens = list(list(enc_childrens[6].children())[0].children())[:-1] + list(list(list(enc_childrens[6].children())[0].children())[-1].children()) + list(list(enc_childrens[6].children())[1].children())
    model.encoder_4_1_conv.load_state_dict(enc_block4_childrens[0].state_dict())
    model.encoder_4_2_batchnorm.load_state_dict(enc_block4_childrens[1].state_dict())
    model.encoder_4_4_conv.load_state_dict(enc_block4_childrens[3].state_dict())
    model.encoder_4_5_batchnorm.load_state_dict(enc_block4_childrens[4].state_dict())
    model.encoder_4_6_downsample.load_state_dict(enc_block4_childrens[5].state_dict())
    model.encoder_4_7_batchnorm.load_state_dict(enc_block4_childrens[6].state_dict())
    model.encoder_4_9_conv.load_state_dict(enc_block4_childrens[7].state_dict())
    model.encoder_4_10_batchnorm.load_state_dict(enc_block4_childrens[8].state_dict())
    model.encoder_4_12_conv.load_state_dict(enc_block4_childrens[10].state_dict())
    model.encoder_4_13_batchnorm.load_state_dict(enc_block4_childrens[11].state_dict())

    enc_block5_childrens = list(list(enc_childrens[7].children())[0].children())[:-1] + list(list(list(enc_childrens[7].children())[0].children())[-1].children()) + list(list(enc_childrens[7].children())[1].children())
    model.encoder_5_1_conv.load_state_dict(enc_block5_childrens[0].state_dict())
    model.encoder_5_2_batchnorm.load_state_dict(enc_block5_childrens[1].state_dict())
    model.encoder_5_4_conv.load_state_dict(enc_block5_childrens[3].state_dict())
    model.encoder_5_5_batchnorm.load_state_dict(enc_block5_childrens[4].state_dict())
    model.encoder_5_6_downsample.load_state_dict(enc_block5_childrens[5].state_dict())
    model.encoder_5_7_batchnorm.load_state_dict(enc_block5_childrens[6].state_dict())
    model.encoder_5_9_conv.load_state_dict(enc_block5_childrens[7].state_dict())
    model.encoder_5_10_batchnorm.load_state_dict(enc_block5_childrens[8].state_dict())
    model.encoder_5_12_conv.load_state_dict(enc_block5_childrens[10].state_dict())
    model.encoder_5_13_batchnorm.load_state_dict(enc_block5_childrens[11].state_dict())

    dec_childrens = list(list(model_smp.decoder.children())[1].children())
    dec_block1_childrens = list(list(dec_childrens[0].children())[0].children()) + list(list(dec_childrens[0].children())[2].children())
    model.decoder_1_1_conv.load_state_dict(dec_block1_childrens[0].state_dict())
    model.decoder_1_2_batchnorm.load_state_dict(dec_block1_childrens[1].state_dict())
    model.decoder_1_4_conv.load_state_dict(dec_block1_childrens[3].state_dict())
    model.decoder_1_5_batchnorm.load_state_dict(dec_block1_childrens[4].state_dict())

    dec_block2_childrens = list(list(dec_childrens[1].children())[0].children()) + list(list(dec_childrens[1].children())[2].children())
    model.decoder_2_1_conv.load_state_dict(dec_block2_childrens[0].state_dict())
    model.decoder_2_2_batchnorm.load_state_dict(dec_block2_childrens[1].state_dict())
    model.decoder_2_4_conv.load_state_dict(dec_block2_childrens[3].state_dict())
    model.decoder_2_5_batchnorm.load_state_dict(dec_block2_childrens[4].state_dict())

    dec_block3_childrens = list(list(dec_childrens[2].children())[0].children()) + list(list(dec_childrens[2].children())[2].children())
    model.decoder_3_1_conv.load_state_dict(dec_block3_childrens[0].state_dict())
    model.decoder_3_2_batchnorm.load_state_dict(dec_block3_childrens[1].state_dict())
    model.decoder_3_4_conv.load_state_dict(dec_block3_childrens[3].state_dict())
    model.decoder_3_5_batchnorm.load_state_dict(dec_block3_childrens[4].state_dict())

    dec_block4_childrens = list(list(dec_childrens[3].children())[0].children()) + list(list(dec_childrens[3].children())[2].children())
    model.decoder_4_1_conv.load_state_dict(dec_block4_childrens[0].state_dict())
    model.decoder_4_2_batchnorm.load_state_dict(dec_block4_childrens[1].state_dict())
    model.decoder_4_4_conv.load_state_dict(dec_block4_childrens[3].state_dict())
    model.decoder_4_5_batchnorm.load_state_dict(dec_block4_childrens[4].state_dict())

    dec_block5_childrens = list(list(dec_childrens[4].children())[0].children()) + list(list(dec_childrens[4].children())[2].children())
    model.decoder_5_1_conv.load_state_dict(dec_block5_childrens[0].state_dict())
    model.decoder_5_2_batchnorm.load_state_dict(dec_block5_childrens[1].state_dict())
    model.decoder_5_4_conv.load_state_dict(dec_block5_childrens[3].state_dict())
    model.decoder_5_5_batchnorm.load_state_dict(dec_block5_childrens[4].state_dict())

    model.decoder_6_1_conv.load_state_dict(list(model_smp.segmentation_head.children())[0].state_dict())

def transfer_weights(model_smp, encoder):

    if (encoder == "resnet18"):
        model = Unet_18(1,3,True)
        transfer_weights_resnet18(model, model_smp)

    if (encoder == "resnet18_relu"):
        model = Unet_18_relu(1,3,True)
        transfer_weights_resnet18(model, model_smp)

    if (encoder == "resnet18_conv"):
        model = Unet_18_conv(1,3,True)
        transfer_weights_resnet18(model, model_smp)

    if (encoder == "resnet18_rep2"):
        model = Unet_18_rep2(1,3,True)
        transfer_weights_resnet18(model, model_smp)

    if (encoder == "vgg19_bn"):
        model = Unet_vgg19(1,3,True)
        transfer_weights_vgg19bn(model, model_smp)

    return model

def compress_weights_vgg19bn(model, model_adapt):

    #first block
    model.encoder_1_1_conv.load_state_dict(model_adapt.encoder_1_1_conv.state_dict())
    model.encoder_1_4_conv.load_state_dict(model_adapt.encoder_1_4_conv.state_dict())
    compress_conv(model.encoder_1_1_conv,model_adapt.adapter_1_2_conv,model_adapt.adapter_1_1_batchnorm,model_adapt.encoder_1_2_batchnorm)
    compress_conv(model.encoder_1_4_conv,model_adapt.adapter_2_2_conv,model_adapt.adapter_2_1_batchnorm,model_adapt.encoder_1_5_batchnorm)

    #second block
    model.encoder_2_1_conv.load_state_dict(model_adapt.encoder_2_1_conv.state_dict())
    model.encoder_2_4_conv.load_state_dict(model_adapt.encoder_2_4_conv.state_dict())
    compress_conv(model.encoder_2_1_conv,model_adapt.adapter_3_2_conv,model_adapt.adapter_3_1_batchnorm,model_adapt.encoder_2_2_batchnorm)
    compress_conv(model.encoder_2_4_conv,model_adapt.adapter_4_2_conv,model_adapt.adapter_4_1_batchnorm,model_adapt.encoder_2_5_batchnorm)
    
    #third block
    model.encoder_3_1_conv.load_state_dict(model_adapt.encoder_3_1_conv.state_dict())
    model.encoder_3_4_conv.load_state_dict(model_adapt.encoder_3_4_conv.state_dict())
    model.encoder_3_7_conv.load_state_dict(model_adapt.encoder_3_7_conv.state_dict())
    model.encoder_3_10_conv.load_state_dict(model_adapt.encoder_3_10_conv.state_dict())
    compress_conv(model.encoder_3_1_conv,model_adapt.adapter_5_2_conv,model_adapt.adapter_5_1_batchnorm,model_adapt.encoder_3_2_batchnorm)
    compress_conv(model.encoder_3_4_conv,model_adapt.adapter_6_2_conv,model_adapt.adapter_6_1_batchnorm,model_adapt.encoder_3_5_batchnorm)
    compress_conv(model.encoder_3_7_conv,model_adapt.adapter_7_2_conv,model_adapt.adapter_7_1_batchnorm,model_adapt.encoder_3_8_batchnorm)
    compress_conv(model.encoder_3_10_conv,model_adapt.adapter_8_2_conv,model_adapt.adapter_8_1_batchnorm,model_adapt.encoder_3_11_batchnorm)
    
    #fourth block
    model.encoder_4_1_conv.load_state_dict(model_adapt.encoder_4_1_conv.state_dict())
    model.encoder_4_4_conv.load_state_dict(model_adapt.encoder_4_4_conv.state_dict())
    model.encoder_4_7_conv.load_state_dict(model_adapt.encoder_4_7_conv.state_dict())
    model.encoder_4_10_conv.load_state_dict(model_adapt.encoder_4_10_conv.state_dict())
    compress_conv(model.encoder_4_1_conv,model_adapt.adapter_9_2_conv,model_adapt.adapter_9_1_batchnorm,model_adapt.encoder_4_2_batchnorm)
    compress_conv(model.encoder_4_4_conv,model_adapt.adapter_10_2_conv,model_adapt.adapter_10_1_batchnorm,model_adapt.encoder_4_5_batchnorm)
    compress_conv(model.encoder_4_7_conv,model_adapt.adapter_11_2_conv,model_adapt.adapter_11_1_batchnorm,model_adapt.encoder_4_8_batchnorm)
    compress_conv(model.encoder_4_10_conv,model_adapt.adapter_12_2_conv,model_adapt.adapter_12_1_batchnorm,model_adapt.encoder_4_11_batchnorm)

    #fifth block
    model.encoder_5_1_conv.load_state_dict(model_adapt.encoder_5_1_conv.state_dict())
    model.encoder_5_4_conv.load_state_dict(model_adapt.encoder_5_4_conv.state_dict())
    model.encoder_5_7_conv.load_state_dict(model_adapt.encoder_5_7_conv.state_dict())
    model.encoder_5_10_conv.load_state_dict(model_adapt.encoder_5_10_conv.state_dict())
    compress_conv(model.encoder_5_1_conv,model_adapt.adapter_13_2_conv,model_adapt.adapter_13_1_batchnorm,model_adapt.encoder_5_2_batchnorm)
    compress_conv(model.encoder_5_4_conv,model_adapt.adapter_14_2_conv,model_adapt.adapter_14_1_batchnorm,model_adapt.encoder_5_5_batchnorm)
    compress_conv(model.encoder_5_7_conv,model_adapt.adapter_15_2_conv,model_adapt.adapter_15_1_batchnorm,model_adapt.encoder_5_8_batchnorm)
    compress_conv(model.encoder_5_10_conv,model_adapt.adapter_16_2_conv,model_adapt.adapter_16_1_batchnorm,model_adapt.encoder_5_11_batchnorm)

    #center block
    model.decoder_1_1_conv.load_state_dict(model_adapt.decoder_1_1_conv.state_dict())
    model.decoder_1_4_conv.load_state_dict(model_adapt.decoder_1_4_conv.state_dict())
    compress_conv(model.decoder_1_1_conv,model_adapt.adapter_17_2_conv,model_adapt.adapter_17_1_batchnorm,model_adapt.decoder_1_2_batchnorm)
    compress_conv(model.decoder_1_4_conv,model_adapt.adapter_18_2_conv,model_adapt.adapter_18_1_batchnorm,model_adapt.decoder_1_5_batchnorm)

    #first decoder block
    model.decoder_2_1_conv.load_state_dict(model_adapt.decoder_2_1_conv.state_dict())
    model.decoder_2_4_conv.load_state_dict(model_adapt.decoder_2_4_conv.state_dict())
    compress_conv(model.decoder_2_1_conv,model_adapt.adapter_19_2_conv,model_adapt.adapter_19_1_batchnorm,model_adapt.decoder_2_2_batchnorm)
    compress_conv(model.decoder_2_4_conv,model_adapt.adapter_20_2_conv,model_adapt.adapter_20_1_batchnorm,model_adapt.decoder_2_5_batchnorm)

    #second decoder block
    model.decoder_3_1_conv.load_state_dict(model_adapt.decoder_3_1_conv.state_dict())
    model.decoder_3_4_conv.load_state_dict(model_adapt.decoder_3_4_conv.state_dict())
    compress_conv(model.decoder_3_1_conv,model_adapt.adapter_21_2_conv,model_adapt.adapter_21_1_batchnorm,model_adapt.decoder_3_2_batchnorm)
    compress_conv(model.decoder_3_4_conv,model_adapt.adapter_22_2_conv,model_adapt.adapter_22_1_batchnorm,model_adapt.decoder_3_5_batchnorm)

    #third decoder block
    model.decoder_4_1_conv.load_state_dict(model_adapt.decoder_4_1_conv.state_dict())
    model.decoder_4_4_conv.load_state_dict(model_adapt.decoder_4_4_conv.state_dict())
    compress_conv(model.decoder_4_1_conv,model_adapt.adapter_23_2_conv,model_adapt.adapter_23_1_batchnorm,model_adapt.decoder_4_2_batchnorm)
    compress_conv(model.decoder_4_4_conv,model_adapt.adapter_24_2_conv,model_adapt.adapter_24_1_batchnorm,model_adapt.decoder_4_5_batchnorm)

    #fourth decoder block
    model.decoder_5_1_conv.load_state_dict(model_adapt.decoder_5_1_conv.state_dict())
    model.decoder_5_4_conv.load_state_dict(model_adapt.decoder_5_4_conv.state_dict())
    compress_conv(model.decoder_5_1_conv,model_adapt.adapter_25_2_conv,model_adapt.adapter_25_1_batchnorm,model_adapt.decoder_5_2_batchnorm)
    compress_conv(model.decoder_5_4_conv,model_adapt.adapter_26_2_conv,model_adapt.adapter_26_1_batchnorm,model_adapt.decoder_5_5_batchnorm)

    #fifth decoder block and segmentation head
    model.decoder_6_1_conv.load_state_dict(model_adapt.decoder_6_1_conv.state_dict())
    model.decoder_6_4_conv.load_state_dict(model_adapt.decoder_6_4_conv.state_dict())
    model.decoder_7_1_conv.load_state_dict(model_adapt.decoder_7_1_conv.state_dict())
    compress_conv(model.decoder_6_1_conv,model_adapt.adapter_27_2_conv,model_adapt.adapter_27_1_batchnorm,model_adapt.decoder_6_2_batchnorm)
    compress_conv(model.decoder_6_4_conv,model_adapt.adapter_28_2_conv,model_adapt.adapter_28_1_batchnorm,model_adapt.decoder_6_5_batchnorm)

def compress_weights_resnet18(model, model_adapt):

    model.encoder_1_1_conv.load_state_dict(model_adapt.encoder_1_1_conv.state_dict())
    model.encoder_1_4_maxpool.load_state_dict(model_adapt.encoder_1_4_maxpool.state_dict())
    compress_conv(model.encoder_1_1_conv,model_adapt.adapter_1_2_conv,model_adapt.adapter_1_1_batchnorm,model_adapt.encoder_1_2_batchnorm)

    model.encoder_2_1_conv.load_state_dict(model_adapt.encoder_2_1_conv.state_dict())
    model.encoder_2_4_conv.load_state_dict(model_adapt.encoder_2_4_conv.state_dict()) 
    model.encoder_2_7_conv.load_state_dict(model_adapt.encoder_2_7_conv.state_dict())
    model.encoder_2_10_conv.load_state_dict(model_adapt.encoder_2_10_conv.state_dict())
    compress_conv(model.encoder_2_1_conv,model_adapt.adapter_2_2_conv,model_adapt.adapter_2_1_batchnorm,model_adapt.encoder_2_2_batchnorm)
    compress_conv(model.encoder_2_4_conv,model_adapt.adapter_3_2_conv,model_adapt.adapter_3_1_batchnorm,model_adapt.encoder_2_5_batchnorm)
    compress_conv(model.encoder_2_7_conv,model_adapt.adapter_4_2_conv,model_adapt.adapter_4_1_batchnorm,model_adapt.encoder_2_8_batchnorm)
    compress_conv(model.encoder_2_10_conv,model_adapt.adapter_5_2_conv,model_adapt.adapter_5_1_batchnorm,model_adapt.encoder_2_11_batchnorm)
                                        
    model.encoder_3_1_conv.load_state_dict(model_adapt.encoder_3_1_conv.state_dict())
    model.encoder_3_4_conv.load_state_dict(model_adapt.encoder_3_4_conv.state_dict())
    model.encoder_3_6_downsample.load_state_dict(model_adapt.encoder_3_6_downsample.state_dict())
    model.encoder_3_7_batchnorm.load_state_dict(model_adapt.encoder_3_7_batchnorm.state_dict())
    model.encoder_3_9_conv.load_state_dict(model_adapt.encoder_3_9_conv.state_dict())
    model.encoder_3_12_conv.load_state_dict(model_adapt.encoder_3_12_conv.state_dict())
    compress_conv(model.encoder_3_1_conv,model_adapt.adapter_6_2_conv,model_adapt.adapter_6_1_batchnorm,model_adapt.encoder_3_2_batchnorm)
    compress_conv(model.encoder_3_4_conv,model_adapt.adapter_7_2_conv,model_adapt.adapter_7_1_batchnorm,model_adapt.encoder_3_5_batchnorm)
    compress_conv(model.encoder_3_9_conv,model_adapt.adapter_8_2_conv,model_adapt.adapter_8_1_batchnorm,model_adapt.encoder_3_10_batchnorm)
    compress_conv(model.encoder_3_12_conv,model_adapt.adapter_9_2_conv,model_adapt.adapter_9_1_batchnorm,model_adapt.encoder_3_13_batchnorm)

    model.encoder_4_1_conv.load_state_dict(model_adapt.encoder_4_1_conv.state_dict())
    model.encoder_4_4_conv.load_state_dict(model_adapt.encoder_4_4_conv.state_dict())
    model.encoder_4_6_downsample.load_state_dict(model_adapt.encoder_4_6_downsample.state_dict())
    model.encoder_4_7_batchnorm.load_state_dict(model_adapt.encoder_4_7_batchnorm.state_dict())
    model.encoder_4_9_conv.load_state_dict(model_adapt.encoder_4_9_conv.state_dict())
    model.encoder_4_12_conv.load_state_dict(model_adapt.encoder_4_12_conv.state_dict())
    compress_conv(model.encoder_4_1_conv,model_adapt.adapter_10_2_conv,model_adapt.adapter_10_1_batchnorm,model_adapt.encoder_4_2_batchnorm)
    compress_conv(model.encoder_4_4_conv,model_adapt.adapter_11_2_conv,model_adapt.adapter_11_1_batchnorm,model_adapt.encoder_4_5_batchnorm)
    compress_conv(model.encoder_4_9_conv,model_adapt.adapter_12_2_conv,model_adapt.adapter_12_1_batchnorm,model_adapt.encoder_4_10_batchnorm)
    compress_conv(model.encoder_4_12_conv,model_adapt.adapter_13_2_conv,model_adapt.adapter_13_1_batchnorm,model_adapt.encoder_4_13_batchnorm)

    model.encoder_5_1_conv.load_state_dict(model_adapt.encoder_5_1_conv.state_dict())
    model.encoder_5_4_conv.load_state_dict(model_adapt.encoder_5_4_conv.state_dict())
    model.encoder_5_6_downsample.load_state_dict(model_adapt.encoder_5_6_downsample.state_dict())
    model.encoder_5_7_batchnorm.load_state_dict(model_adapt.encoder_5_7_batchnorm.state_dict())
    model.encoder_5_9_conv.load_state_dict(model_adapt.encoder_5_9_conv.state_dict())
    model.encoder_5_12_conv.load_state_dict(model_adapt.encoder_5_12_conv.state_dict())
    compress_conv(model.encoder_5_1_conv,model_adapt.adapter_14_2_conv,model_adapt.adapter_14_1_batchnorm,model_adapt.encoder_5_2_batchnorm)
    compress_conv(model.encoder_5_4_conv,model_adapt.adapter_15_2_conv,model_adapt.adapter_15_1_batchnorm,model_adapt.encoder_5_5_batchnorm)
    compress_conv(model.encoder_5_9_conv,model_adapt.adapter_16_2_conv,model_adapt.adapter_16_1_batchnorm,model_adapt.encoder_5_10_batchnorm)
    compress_conv(model.encoder_5_12_conv,model_adapt.adapter_17_2_conv,model_adapt.adapter_17_1_batchnorm,model_adapt.encoder_5_13_batchnorm)

    model.decoder_1_1_conv.load_state_dict(model_adapt.decoder_1_1_conv.state_dict())
    model.decoder_1_4_conv.load_state_dict(model_adapt.decoder_1_4_conv.state_dict())
    compress_conv(model.decoder_1_1_conv,model_adapt.adapter_18_2_conv,model_adapt.adapter_18_1_batchnorm,model_adapt.decoder_1_2_batchnorm)
    compress_conv(model.decoder_1_4_conv,model_adapt.adapter_19_2_conv,model_adapt.adapter_19_1_batchnorm,model_adapt.decoder_1_5_batchnorm)

    model.decoder_2_1_conv.load_state_dict(model_adapt.decoder_2_1_conv.state_dict())
    model.decoder_2_4_conv.load_state_dict(model_adapt.decoder_2_4_conv.state_dict())
    compress_conv(model.decoder_2_1_conv,model_adapt.adapter_20_2_conv,model_adapt.adapter_20_1_batchnorm,model_adapt.decoder_2_2_batchnorm)
    compress_conv(model.decoder_2_4_conv,model_adapt.adapter_21_2_conv,model_adapt.adapter_21_1_batchnorm,model_adapt.decoder_2_5_batchnorm)

    model.decoder_3_1_conv.load_state_dict(model_adapt.decoder_3_1_conv.state_dict())
    model.decoder_3_4_conv.load_state_dict(model_adapt.decoder_3_4_conv.state_dict())
    compress_conv(model.decoder_3_1_conv,model_adapt.adapter_22_2_conv,model_adapt.adapter_22_1_batchnorm,model_adapt.decoder_3_2_batchnorm)
    compress_conv(model.decoder_3_4_conv,model_adapt.adapter_23_2_conv,model_adapt.adapter_23_1_batchnorm,model_adapt.decoder_3_5_batchnorm)

    model.decoder_4_1_conv.load_state_dict(model_adapt.decoder_4_1_conv.state_dict())
    model.decoder_4_4_conv.load_state_dict(model_adapt.decoder_4_4_conv.state_dict())
    compress_conv(model.decoder_4_1_conv,model_adapt.adapter_24_2_conv,model_adapt.adapter_24_1_batchnorm,model_adapt.decoder_4_2_batchnorm)
    compress_conv(model.decoder_4_4_conv,model_adapt.adapter_25_2_conv,model_adapt.adapter_25_1_batchnorm,model_adapt.decoder_4_5_batchnorm)

    model.decoder_5_1_conv.load_state_dict(model_adapt.decoder_5_1_conv.state_dict())
    model.decoder_5_4_conv.load_state_dict(model_adapt.decoder_5_4_conv.state_dict())
    compress_conv(model.decoder_5_1_conv,model_adapt.adapter_26_2_conv,model_adapt.adapter_26_1_batchnorm,model_adapt.decoder_5_2_batchnorm)
    compress_conv(model.decoder_5_4_conv,model_adapt.adapter_27_2_conv,model_adapt.adapter_27_1_batchnorm,model_adapt.decoder_5_5_batchnorm)

    model.decoder_6_1_conv.load_state_dict(model_adapt.decoder_6_1_conv.state_dict())

def compress_weights(model_adapt, encoder):

    model_adapt.to(device = "cpu")
    model_adapt.eval()

    if (encoder == "resnet18"):
        model = Unet_18_fuse(1,3).to(device="cpu")
        model.eval()
        compress_weights_resnet18(model, model_adapt)

    if (encoder == "vgg19_bn"):
        model = Unet_vgg19_fuse(1,3).to(device="cpu")
        model.eval()
        compress_weights_vgg19bn(model, model_adapt)

    return model

def compress_conv(convbias,convbias_adapt,bnorm_adapt,bnorm):

    #-----------------------------------------------
    #Step 1: linearizing 1x1 convolution and adapter
    #-----------------------------------------------

    #getting parameters of batchnorm
    running_std = torch.sqrt(bnorm_adapt.running_var)
    running_mean = bnorm_adapt.running_mean
    gamma = bnorm_adapt.weight
    beta = bnorm_adapt.bias
    
    #getting shape features
    n_features = running_std.size()[0]
    
    #if there are no biases generating zero arrays
    if (convbias.bias == None): convbias.bias = nn.Parameter(torch.zeros(n_features),requires_grad=True)
    if (convbias_adapt.bias == None): convbias_adapt.bias = nn.Parameter(torch.zeros(n_features),requires_grad=True)

    #parameters of convolution
    weight = convbias_adapt.weight.data
    bias = convbias_adapt.bias.data
    
    #new weights of 1x1 convolution
    newweight = torch.eye(n_features,n_features) + weight.view(n_features,n_features)*gamma/running_std
    newbias = bias + torch.einsum('cd,d->c',weight.view(n_features,n_features),beta - gamma*running_mean/running_std)

    #----------------------------------------------
    #Step 2: fusing 1x1 convolution and batchnorm
    #----------------------------------------------
    
    #getting parameters of batchnorm
    running_std = torch.sqrt(bnorm.running_var)
    running_mean = bnorm.running_mean
    gamma = bnorm.weight.data
    beta = bnorm.bias.data

    # Compute the new weights
    scale = gamma / running_std
    newweight = newweight.view(n_features,n_features,1,1) * scale.reshape([-1, 1, 1, 1])
    newbias = beta + (newbias - running_mean) * scale

    #defining new weight and bias of 1x1 convolution
    '''
    newweight = newweight.view(n_features,n_features)*gamma/running_std
    newbias = (newbias - running_mean)/running_std*gamma + beta
    '''
    
    #----------------------------------------------
    #Step 3: fusing 3x3 and 1x1 convolution
    #----------------------------------------------
    
    #adjusting shape of 1x1
    newweight = newweight.view(n_features,n_features,1,1)

    # Initialize fused weights and biases
    weight_fuse = torch.zeros_like(convbias.weight.data)
    bias_fuse = torch.zeros_like(convbias.bias.data)

    # Efficient computation of the fused weights using broadcasting
    # W_1x1 is reshaped to (C_out, C_mid) and W_3x3 remains (C_mid, C_in, 3, 3)
    # The result will be broadcast to (C_out, C_mid, C_in, 3, 3) and then summed along axis=1
    weight_fuse = torch.sum(newweight[:, :, 0, 0].unsqueeze(2).unsqueeze(3).unsqueeze(4) * convbias.weight.data.unsqueeze(0), dim=1)

    # Efficient computation of the fused biases
    bias_fuse = newbias + torch.sum(newweight[:, :, 0, 0] * convbias.bias.data, dim=1)

    '''
    # Compute the fused weights
    for o in range(n_features):
        for m in range(n_features):
            weight_fuse[o, :, :, :] += newweight[o, m, 0, 0] * convbias.weight.data[m, :, :, :]

    # Compute the fused biases
    for o in range(n_features):
        bias_fuse[o] = newbias[o] + torch.sum(newweight[o, :, 0, 0] * convbias.bias.data)
    '''

    #----------------------------------------------
    #Step 4: updating weights and biases
    #----------------------------------------------

    #updating weight and bias
    convbias.weight.data = weight_fuse
    convbias.bias.data = bias_fuse

def remove_adapters(model_adapt, nadapters, encoder, method):

    if encoder == "resnet18":
        adapters_bnorm = [model_adapt.adapter_1_1_batchnorm,model_adapt.adapter_2_1_batchnorm,model_adapt.adapter_3_1_batchnorm,
                    model_adapt.adapter_4_1_batchnorm,model_adapt.adapter_5_1_batchnorm,model_adapt.adapter_6_1_batchnorm,
                    model_adapt.adapter_7_1_batchnorm,model_adapt.adapter_8_1_batchnorm,model_adapt.adapter_9_1_batchnorm,
                    model_adapt.adapter_10_1_batchnorm,model_adapt.adapter_11_1_batchnorm,model_adapt.adapter_12_1_batchnorm,
                    model_adapt.adapter_13_1_batchnorm,model_adapt.adapter_14_1_batchnorm,model_adapt.adapter_15_1_batchnorm,
                    model_adapt.adapter_16_1_batchnorm,model_adapt.adapter_17_1_batchnorm,model_adapt.adapter_18_1_batchnorm,
                    model_adapt.adapter_19_1_batchnorm,model_adapt.adapter_20_1_batchnorm,model_adapt.adapter_21_1_batchnorm,
                    model_adapt.adapter_22_1_batchnorm,model_adapt.adapter_23_1_batchnorm,model_adapt.adapter_24_1_batchnorm,
                    model_adapt.adapter_25_1_batchnorm,model_adapt.adapter_26_1_batchnorm,model_adapt.adapter_27_1_batchnorm]
        adapters = [model_adapt.adapter_1_2_conv,model_adapt.adapter_2_2_conv,model_adapt.adapter_3_2_conv,
                    model_adapt.adapter_4_2_conv,model_adapt.adapter_5_2_conv,model_adapt.adapter_6_2_conv,
                    model_adapt.adapter_7_2_conv,model_adapt.adapter_8_2_conv,model_adapt.adapter_9_2_conv,
                    model_adapt.adapter_10_2_conv,model_adapt.adapter_11_2_conv,model_adapt.adapter_12_2_conv,
                    model_adapt.adapter_13_2_conv,model_adapt.adapter_14_2_conv,model_adapt.adapter_15_2_conv,
                    model_adapt.adapter_16_2_conv,model_adapt.adapter_17_2_conv,model_adapt.adapter_18_2_conv,
                    model_adapt.adapter_19_2_conv,model_adapt.adapter_20_2_conv,model_adapt.adapter_21_2_conv,
                    model_adapt.adapter_22_2_conv,model_adapt.adapter_23_2_conv,model_adapt.adapter_24_2_conv,
                    model_adapt.adapter_25_2_conv,model_adapt.adapter_26_2_conv,model_adapt.adapter_27_2_conv]
    if encoder == "vgg19_bn":
        adapters_bnorm = [model_adapt.adapter_1_1_batchnorm,model_adapt.adapter_2_1_batchnorm,model_adapt.adapter_3_1_batchnorm,
                    model_adapt.adapter_4_1_batchnorm,model_adapt.adapter_5_1_batchnorm,model_adapt.adapter_6_1_batchnorm,
                    model_adapt.adapter_7_1_batchnorm,model_adapt.adapter_8_1_batchnorm,model_adapt.adapter_9_1_batchnorm,
                    model_adapt.adapter_10_1_batchnorm,model_adapt.adapter_11_1_batchnorm,model_adapt.adapter_12_1_batchnorm,
                    model_adapt.adapter_13_1_batchnorm,model_adapt.adapter_14_1_batchnorm,model_adapt.adapter_15_1_batchnorm,
                    model_adapt.adapter_16_1_batchnorm,model_adapt.adapter_17_1_batchnorm,model_adapt.adapter_18_1_batchnorm,
                    model_adapt.adapter_19_1_batchnorm,model_adapt.adapter_20_1_batchnorm,model_adapt.adapter_21_1_batchnorm,
                    model_adapt.adapter_22_1_batchnorm,model_adapt.adapter_23_1_batchnorm,model_adapt.adapter_24_1_batchnorm,
                    model_adapt.adapter_25_1_batchnorm,model_adapt.adapter_26_1_batchnorm,model_adapt.adapter_27_1_batchnorm,model_adapt.adapter_28_1_batchnorm]
        adapters = [model_adapt.adapter_1_2_conv,model_adapt.adapter_2_2_conv,model_adapt.adapter_3_2_conv,
                    model_adapt.adapter_4_2_conv,model_adapt.adapter_5_2_conv,model_adapt.adapter_6_2_conv,
                    model_adapt.adapter_7_2_conv,model_adapt.adapter_8_2_conv,model_adapt.adapter_9_2_conv,
                    model_adapt.adapter_10_2_conv,model_adapt.adapter_11_2_conv,model_adapt.adapter_12_2_conv,
                    model_adapt.adapter_13_2_conv,model_adapt.adapter_14_2_conv,model_adapt.adapter_15_2_conv,
                    model_adapt.adapter_16_2_conv,model_adapt.adapter_17_2_conv,model_adapt.adapter_18_2_conv,
                    model_adapt.adapter_19_2_conv,model_adapt.adapter_20_2_conv,model_adapt.adapter_21_2_conv,
                    model_adapt.adapter_22_2_conv,model_adapt.adapter_23_2_conv,model_adapt.adapter_24_2_conv,
                    model_adapt.adapter_25_2_conv,model_adapt.adapter_26_2_conv,model_adapt.adapter_27_2_conv,model_adapt.adapter_28_2_conv]

    #empty lists
    wnorm_nparams = []
    wnorm = []
    nparams = []
    sizeadapt = []

    for iadapt in range(len(adapters)):

        #getting norm of weights
        convnorm = np.linalg.norm(adapters[iadapt].weight.data.cpu())
        if (adapters[iadapt].bias is not None): biasnorm = np.linalg.norm(adapters[iadapt].bias.data.cpu())
        else: biasnorm = 0.
        gammanorm = np.linalg.norm(adapters_bnorm[iadapt].weight.data.cpu())
        betanorm = np.linalg.norm(adapters_bnorm[iadapt].bias.data.cpu())
        totnorm = np.sqrt(convnorm**2 + biasnorm**2 + betanorm**2 + gammanorm**2)

        #getting number of parameters
        adaptparams = sum([len(torch.flatten(p)) for p in adapters[iadapt].parameters()])
        bnormparams = sum([len(torch.flatten(p)) for p in adapters_bnorm[iadapt].parameters()])
        totparams = adaptparams + bnormparams 

        #getting size
        torch.save(adapters[iadapt],'adapt_temp.pt')
        stats_adapt = os.stat('adapt_temp.pt')
        os.remove('adapt_temp.pt')
        torch.save(adapters_bnorm[iadapt],'adapt_temp.pt')
        stats_bnorm = os.stat('adapt_temp.pt')
        os.remove('adapt_temp.pt')
        sizetot = stats_adapt.st_size + stats_bnorm.st_size

        #appending lists
        wnorm.append(totnorm)
        nparams.append(totparams)
        wnorm_nparams.append(totnorm/totparams)
        sizeadapt.append(sizetot/(1024 * 1024))

    #sorting
    if (method == "wnorm_nparams"): 
        arrsort = wnorm_nparams
        isorted = np.argsort(arrsort)
    if (method == "wnorm"): 
        arrsort = wnorm
        isorted = np.argsort(arrsort)
    if (method == "nparams"): 
        arrsort = nparams
        isorted = np.argsort(arrsort)
    if (method == "forward"): 
        isorted = list(range(len(wnorm)))
    if (method == "backward"): 
        isorted = list(range(len(wnorm)))
        isorted.reverse()

    adapters = [adapters[i] for i in isorted]
    adapters_bnorm = [adapters_bnorm[i] for i in isorted]

    #selecting and remove adapters
    wnorm = np.array(wnorm)[isorted]
    nparams = np.array(nparams)[isorted]
    sizeadapt = np.array(sizeadapt)[isorted]
    if (len(adapters) == nadapters):
        wnorm = 0
        nparams = 0
        sizeadapt = 0
    else:
        wnorm = np.sum(wnorm[nadapters:len(adapters)])
        nparams = np.sum(nparams[nadapters:len(adapters)])
        sizeadapt = np.sum(sizeadapt[nadapters:len(adapters)])

    for iadapt,adapt in enumerate(adapters):
        if (iadapt + 1 > nadapters): break
        model_adapt.init_convweights(adapt)
        model_adapt.init_bnorm(adapters_bnorm[iadapt])

    return model_adapt,wnorm,nparams,sizeadapt,isorted

#testing the function

# batch = torch.rand([32, 1, 224, 608])
# model = Unet_18_fuse(1,3)
# model_adapt = Unet_18(1,3,True)
# model.eval()
# model_adapt.eval()

# pred = model(batch)
# adapt_pred = model_adapt(batch)

# print("Before Initialization: ")
# print(torch.equal(pred,adapt_pred))

# compress_weights_resnet18(model,model_adapt)

# pred = model(batch)
# adapt_pred = model_adapt(batch)

# print("After Initialization: ")
# print(pred[0][0][0][1],adapt_pred[0][0][0][1])
# print(torch.equal(pred,adapt_pred))
# print(torch.min(abs(torch.flatten(pred) - torch.flatten(adapt_pred))))
# print(torch.mean(abs(torch.flatten(pred) - torch.flatten(adapt_pred))))
# print(torch.max(abs(torch.flatten(pred) - torch.flatten(adapt_pred))))

# batch = torch.rand([32, 1, 224, 608])
# model = Unet_vgg19(1,3,True)
# model.eval()

# model_smp = smp.Unet(encoder_name = 'vgg19_bn', classes = 3, activation = None, in_channels = 1, encoder_weights=None)

# model.eval()
# model_smp.eval()

# pred = model(batch)
# smp_pred = model_smp(batch)

# print("Before Initialization: ")
# print(torch.equal(pred,smp_pred))

# transfer_weights_vgg19bn(model,model_smp)

# pred = model(batch)
# smp_pred = model_smp(batch)

# print("After Initialization: ")
# print(pred[0][0][0][0],smp_pred[0][0][0][0])
# print(torch.equal(pred,smp_pred))
