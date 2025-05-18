import model_library
from model_library import *

#check if cuda is available
device = "cpu"
device_id = torch.cuda.current_device()
print("Training on Device: ", device)
print("Device_id: ", device_id)
print("GPU: ",torch.cuda.get_device_name(device_id))

import logging
logging.getLogger('fvcore').setLevel(logging.ERROR)

def perc(x):
    x = np.array(x).astype(float)
    print(np.size(x))
    x = x / x[len(x)-1]
    return x

#-------------------------------------------------------------
#--------------------ADAPTERS IN RESNET-18--------------------
#-------------------------------------------------------------

#modelsdir
modelsdir = "real-moon-models/"
img_test = torch.ones([1,1,512,512])

#resnet-18
params_train_resnet = []
params_perc_resnet = []
flops_arr_resnet = []
baselines = [False,True,False,False,False,False,False]
encs = [False,False,True,True,False,False,False]
decs = [False,False,True,False,True,False,False]
bnorms = [False,False,False,False,False,True,False]
adapts = [False,False,False,False,False,False,True]
normbals = []

for ibal in range(len(baselines)):

    encoder_name = "resnet18"
    baseline = baselines[ibal]
    ftuneenc = encs[ibal]
    ftunedec = decs[ibal]
    ftunebnorm = bnorms[ibal]
    ftuneadapt = adapts[ibal]

    unet_train = saveorload(modelsdir,1,0,0,0,0,0.5,encoder_name,"adam","BalancedCCE",baseline,ftuneenc,ftunedec,ftunebnorm,ftuneadapt)
    model_train(unet_train, encoder_name, baseline, ftuneenc, ftunedec, ftunebnorm, ftuneadapt)
    non_frozen_parameters = sum([len(torch.flatten(p)) for p in unet_train.parameters() if p.requires_grad])
    if (ibal==0): non_frozen_parameters = 0
    tot_parameters = sum([len(torch.flatten(p)) for p in unet_train.parameters()])
    params_train_resnet.append(non_frozen_parameters / 1e6)

    if (ibal == 0): params_perc_resnet.append(non_frozen_parameters / tot_parameters * 100)
    if (ibal == 1): params_perc_resnet.append(non_frozen_parameters / tot_parameters * 100)
    if (ibal > 1 and ibal < 6): params_perc_resnet.append(non_frozen_parameters / params_train_resnet[1] / 1e6 * 100)
    if (ibal >= 6): params_perc_resnet.append(non_frozen_parameters / tot_parameters * 100)

    #slight modification for flop counter
    if (ibal < 6):
        unet_train = saveorload(modelsdir,1,0,0,0,0,0.5,encoder_name,"adam","BalancedCCE",True, False, False, False, False)
        model_train(unet_train, encoder_name, True, False, False, False, False)
    if (ibal >=6):
        unet_train = saveorload(modelsdir,1,0,0,0,0,0.5,encoder_name,"adam","BalancedCCE",False, True, True, True, True)
        model_train(unet_train, encoder_name, False, True, True, True, True)
    if (ibal == 6): 
        unet_load,wnorm,nparams,size,isorted = remove_adapters(unet_train,9,encoder_name,"wnorm_nparams")
        used_adapters = [isortadapt + 1 for isortadapt in isorted][9:]
        used_adapters.sort()
        print("Adapters used:", used_adapters)
        unet_train = Unet_18_ranked(1,3,True)
        unet_train.load_state_dict(unet_load.state_dict())

    #flop counter
    flops = FlopCountAnalysis(unet_train, img_test)
    nflops = flops.total() / 1e9
    flops_arr_resnet.append(nflops)

print("----------------------------------------------------")
print("Adapters in Resnet18:")
print("Trainable Parameters (M): ",params_train_resnet)
print("Trainable Parameters (%s): " % ("%"),params_perc_resnet) 
print("FLOPs (G):",flops_arr_resnet)
print("----------------------------------------------------")

#-------------------------------------------------------------
#--------------------ADAPTERS IN VGG-19 BN--------------------
#-------------------------------------------------------------

#modelsdir
modelsdir = "real-moon-models/"

#resnet-18
params_train_vgg = []
params_perc_vgg = []
flops_arr_vgg = []
baselines = [False,True,False,False,False,False,False]
encs = [False,False,True,True,False,False,False]
decs = [False,False,True,False,True,False,False]
bnorms = [False,False,False,False,False,True,False]
adapts = [False,False,False,False,False,False,True]
normbals = []

for ibal in range(len(baselines)):

    encoder_name = "vgg19_bn"
    baseline = baselines[ibal]
    ftuneenc = encs[ibal]
    ftunedec = decs[ibal]
    ftunebnorm = bnorms[ibal]
    ftuneadapt = adapts[ibal]

    unet_train = saveorload(modelsdir,1,0,0,0,0,0.5,encoder_name,"adam","BalancedCCE",baseline,ftuneenc,ftunedec,ftunebnorm,ftuneadapt)
    model_train(unet_train, encoder_name, baseline, ftuneenc, ftunedec, ftunebnorm, ftuneadapt)
    non_frozen_parameters = sum([len(torch.flatten(p)) for p in unet_train.parameters() if p.requires_grad])
    if (ibal==0): non_frozen_parameters = 0
    tot_parameters = sum([len(torch.flatten(p)) for p in unet_train.parameters()])
    params_train_vgg.append(non_frozen_parameters / 1e6)

    if (ibal == 0): params_perc_vgg.append(non_frozen_parameters / tot_parameters * 100)
    if (ibal == 1): params_perc_vgg.append(non_frozen_parameters / tot_parameters * 100)
    if (ibal > 1 and ibal < 6): params_perc_vgg.append(non_frozen_parameters / params_train_vgg[1] / 1e6 * 100)
    if (ibal >= 6): params_perc_vgg.append(non_frozen_parameters / tot_parameters * 100)

    #slight modification for flop counter
    if (ibal < 6):
        unet_train = saveorload(modelsdir,1,0,0,0,0,0.5,encoder_name,"adam","BalancedCCE",True, False, False, False, False)
        model_train(unet_train, encoder_name, True, False, False, False, False)
    if (ibal >=6):
        unet_train = saveorload(modelsdir,1,0,0,0,0,0.5,encoder_name,"adam","BalancedCCE",False, True, True, True, True)
        model_train(unet_train, encoder_name, False, True, True, True, True)
    #if (ibal == 6): unet_train = compress_weights(unet_train, encoder_name)
    if (ibal == 6): 
        unet_load,wnorm,nparams,size,isorted = remove_adapters(unet_train,11,encoder_name,"wnorm_nparams")
        used_adapters = [isortadapt + 1 for isortadapt in isorted][11:]
        used_adapters.sort()
        print("Adapters used:", used_adapters)
        unet_train = Unet_vgg19_ranked(1,3,True)
        unet_train.load_state_dict(unet_load.state_dict())

    #flop counter
    flops = FlopCountAnalysis(unet_train, img_test)
    nflops = flops.total() / 1e9
    flops_arr_vgg.append(nflops)

print("----------------------------------------------------")
print("Adapters in Vgg19-BN:")
print("Trainable Parameters (M): ",params_train_vgg)
print("Trainable Parameters (%s): " % ("%"),params_perc_vgg)
print("FLOPs (G):",flops_arr_vgg)
print("----------------------------------------------------")