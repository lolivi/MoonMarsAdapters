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

'''

# # ------------------------------------------------------------------------------------------
# # PLOTS

balaccs_resnet = [57.93,86.75,92.23,91.67,89.95,85.54,90.82]
balaccs_vgg = [52.94,84.46,91.88,91.94,88.14,86.89,89.38]

fig, ax = plt.subplots(2,2,figsize = (12,5),clear = True)
msize = 150

ax[0][0].scatter(params_train_resnet[0],balaccs_resnet[0], color = "green", label = "Baseline", marker = "*", s = msize)
ax[0][0].scatter(params_train_resnet[1],balaccs_resnet[1], color = "green", label = "Scratch", marker = ">", s = msize)
ax[0][0].scatter(params_train_resnet[2],balaccs_resnet[2], color = "blue", label = "Full finetuning", marker = "s", s = msize)
ax[0][0].scatter(params_train_resnet[3],balaccs_resnet[3], color = "blue", label = "Encoder finetuning", marker = "p", s = msize)
ax[0][0].scatter(params_train_resnet[4],balaccs_resnet[4], color = "blue", label = "Decoder finetuning", marker = "v", s = msize)
ax[0][0].scatter(params_train_resnet[5],balaccs_resnet[5], color = "blue", label = "Batchnorm finetuning", marker = "^", s = msize)
ax[0][0].scatter(params_train_resnet[6],balaccs_resnet[6], color = "red", label = "Adapters", marker = "o", s = msize)
ax[0][0].grid(True, which='major')
ax[0][0].grid(True, which='minor', alpha=0.2)
ax[0][0].set_axisbelow(True)
ax[0][0].minorticks_on()

ax[1][0].scatter(params_train_resnet[0],balaccs_resnet[0], color = "green", label = "Baseline", marker = "*", s = msize)
ax[1][0].scatter(params_train_resnet[1],balaccs_resnet[1], color = "green", label = "Scratch", marker = ">", s = msize)
ax[1][0].scatter(params_train_resnet[2],balaccs_resnet[2], color = "blue", label = "Full finetuning", marker = "s", s = msize)
ax[1][0].scatter(params_train_resnet[3],balaccs_resnet[3], color = "blue", label = "Encoder finetuning", marker = "p", s = msize)
ax[1][0].scatter(params_train_resnet[4],balaccs_resnet[4], color = "blue", label = "Decoder finetuning", marker = "v", s = msize)
ax[1][0].scatter(params_train_resnet[5],balaccs_resnet[5], color = "blue", label = "Batchnorm finetuning", marker = "^", s = msize)
ax[1][0].scatter(params_train_resnet[6],balaccs_resnet[6], color = "red", label = "Adapters", marker = "o", s = msize)
ax[1][0].set_xlabel('ResNet-18 - Number of Trained Parameters [M]')
ax[1][0].grid(True, which='major')
ax[1][0].grid(True, which='minor', alpha=0.2)
ax[1][0].set_axisbelow(True)
ax[1][0].minorticks_on()

# zoom-in / limit the view to different portions of the data
ax[0][0].set_ylim(78, 100)  # outliers only
ax[1][0].set_ylim(50, 60)  # most of the data

# hide the spines between ax and ax2
ax[0][0].spines.bottom.set_visible(False)
ax[1][0].spines.top.set_visible(False)
ax[0][0].xaxis.tick_top()
ax[0][0].tick_params(labeltop=False)  # don't put tick labels at the top
ax[1][0].xaxis.tick_bottom()

#fig.text(0.04, 0.5, 'Test Balanced Accuracy [%]', va='center', rotation='vertical', fontsize=12)

d = .5  # proportion of vertical to horizontal extent of the slanted line
kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
              linestyle="none", color='k', mec='k', mew=1, clip_on=False)
ax[0][0].plot([0, 1], [0, 0], transform=ax[0][0].transAxes, **kwargs)
ax[1][0].plot([0, 1], [1, 1], transform=ax[1][0].transAxes, **kwargs)

fig.text(0.005, 0.5, 'Test Balanced Accuracy [%]', va='center', rotation='vertical', fontsize=10)

ax[0][1].scatter(params_train_vgg[0],balaccs_vgg[0], color = "green", label = "Baseline", marker = "*", s = msize)
ax[0][1].scatter(params_train_vgg[1],balaccs_vgg[1], color = "green", label = "Scratch", marker = ">", s = msize)
ax[0][1].scatter(params_train_vgg[2],balaccs_vgg[2], color = "blue", label = "Full finetuning", marker = "s", s = msize)
ax[0][1].scatter(params_train_vgg[3],balaccs_vgg[3], color = "blue", label = "Encoder finetuning", marker = "p", s = msize)
ax[0][1].scatter(params_train_vgg[4],balaccs_vgg[4], color = "blue", label = "Decoder finetuning", marker = "v", s = msize)
ax[0][1].scatter(params_train_vgg[5],balaccs_vgg[5], color = "blue", label = "Batchnorm finetuning", marker = "^", s = msize)
ax[0][1].scatter(params_train_vgg[6],balaccs_vgg[6], color = "red", label = "Adapters", marker = "o", s = msize)
ax[0][1].grid(True, which='major')
ax[0][1].grid(True, which='minor', alpha=0.2)
ax[0][1].set_axisbelow(True)
ax[0][1].minorticks_on()

ax[1][1].scatter(params_train_vgg[0],balaccs_vgg[0], color = "green", label = "Baseline", marker = "*", s = msize)
ax[1][1].scatter(params_train_vgg[1],balaccs_vgg[1], color = "green", label = "Scratch", marker = ">", s = msize)
ax[1][1].scatter(params_train_vgg[2],balaccs_vgg[2], color = "blue", label = "Full finetuning", marker = "s", s = msize)
ax[1][1].scatter(params_train_vgg[3],balaccs_vgg[3], color = "blue", label = "Encoder finetuning", marker = "p", s = msize)
ax[1][1].scatter(params_train_vgg[4],balaccs_vgg[4], color = "blue", label = "Decoder finetuning", marker = "v", s = msize)
ax[1][1].scatter(params_train_vgg[5],balaccs_vgg[5], color = "blue", label = "Batchnorm finetuning", marker = "^", s = msize)
ax[1][1].scatter(params_train_vgg[6],balaccs_vgg[6], color = "red", label = "Adapters", marker = "o", s = msize)
ax[1][1].set_xlabel('Vgg19-BN - Number of Trained Parameters [M]')
ax[1][1].grid(True, which='major')
ax[1][1].grid(True, which='minor', alpha=0.2)
ax[1][1].set_axisbelow(True)
ax[1][1].minorticks_on()

# zoom-in / limit the view to different portions of the data
ax[0][1].set_ylim(78, 100)  # outliers only
ax[1][1].set_ylim(50, 60)  # most of the data

# hide the spines between ax and ax2
ax[0][1].spines.bottom.set_visible(False)
ax[1][1].spines.top.set_visible(False)
ax[0][1].xaxis.tick_top()
ax[0][1].tick_params(labeltop=False)  # don't put tick labels at the top
ax[1][1].xaxis.tick_bottom()

d = .5  # proportion of vertical to horizontal extent of the slanted line
kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
              linestyle="none", color='k', mec='k', mew=1, clip_on=False)
ax[0][1].plot([0, 1], [0, 0], transform=ax[0][1].transAxes, **kwargs)
ax[1][1].plot([0, 1], [1, 1], transform=ax[1][1].transAxes, **kwargs)

fig.text(0.5, 0.5, 'Test Balanced Accuracy [%]', va='center', rotation='vertical', fontsize=10)

# Create a single legend for both subplots
handles, labels = ax[0][0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=7, bbox_to_anchor=(0.5, 1.05))

# Adjust layout to make space for the legend
fig.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("adapt_params.pdf", bbox_inches='tight')
plt.close('all')

'''

balaccs_resnet = [57.93,86.75,92.23,91.67,89.95,85.54,90.82]
balaccs_vgg = [52.94,84.46,91.88,91.94,88.14,86.89,89.38]

fig, ax = plt.subplots(1,2,figsize = (12,5),clear = True) #16/4
msize = 150

ax[0].scatter(params_train_resnet[0],balaccs_resnet[0], color = "green", label = "Baseline", marker = "*", s = msize)
ax[0].scatter(params_train_resnet[1],balaccs_resnet[1], color = "green", label = "Scratch", marker = ">", s = msize)
ax[0].scatter(params_train_resnet[2],balaccs_resnet[2], color = "blue", label = "Full finetuning", marker = "s", s = msize)
ax[0].scatter(params_train_resnet[3],balaccs_resnet[3], color = "blue", label = "Encoder finetuning", marker = "p", s = msize)
ax[0].scatter(params_train_resnet[4],balaccs_resnet[4], color = "blue", label = "Decoder finetuning", marker = "v", s = msize)
ax[0].scatter(params_train_resnet[5],balaccs_resnet[5], color = "blue", label = "Batchnorm finetuning", marker = "^", s = msize)
ax[0].scatter(params_train_resnet[6],balaccs_resnet[6], color = "red", label = "Adapters", marker = "o", s = msize)
ax[0].set_xlabel('ResNet-18 - Number of Trained Parameters [M]')
ax[0].set_ylabel('Test Balanced Accuracy [%]')
ax[0].grid(True, which='major')
ax[0].grid(True, which='minor', alpha=0.2)
ax[0].set_axisbelow(True)
ax[0].minorticks_on()

ax[1].scatter(params_train_vgg[0],balaccs_vgg[0], color = "green", marker = "*", s = msize)
ax[1].scatter(params_train_vgg[1],balaccs_vgg[1], color = "green", marker = ">", s = msize)
ax[1].scatter(params_train_vgg[2],balaccs_vgg[2], color = "blue", marker = "s", s = msize)
ax[1].scatter(params_train_vgg[3],balaccs_vgg[3], color = "blue", marker = "p", s = msize)
ax[1].scatter(params_train_vgg[4],balaccs_vgg[4], color = "blue", marker = "v", s = msize)
ax[1].scatter(params_train_vgg[5],balaccs_vgg[5], color = "blue",  marker = "^", s = msize)
ax[1].scatter(params_train_vgg[6],balaccs_vgg[6], color = "red", marker = "o", s = msize)
ax[1].set_xlabel('Vgg19-BN - Number of Trained Parameters [M]')
ax[1].set_ylabel('Test Balanced Accuracy [%]')
ax[1].grid(True,which='both')
ax[1].grid(True, which='major')
ax[1].grid(True, which='minor', alpha=0.2)
ax[1].set_axisbelow(True)
ax[1].minorticks_on()

# Create a single legend for both subplots
handles, labels = ax[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=7, bbox_to_anchor=(0.5, 1.05))

# Adjust layout to make space for the legend
fig.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("adapt_params.pdf", bbox_inches='tight')
plt.close('all')

plt.close('all')
gc.collect()

