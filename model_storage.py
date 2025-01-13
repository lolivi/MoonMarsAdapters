import torch
import os

encoder = "efficientnet-b3" #vgg19_bn, resnet18, mobilenet_v2
dataset = "synthetic-moon" #marsdataset3, real-moon, synthetic-moon, ai4mars
adapters = False
bnorm = False
lin = False
modeldir = encoder + "_adam_BalancedCCE_RandomCrop"
if (bnorm): modeldir = modeldir + "_bnorm"
if (adapters): modeldir = modeldir + "_adapter"
if (lin):
    model = dataset + "-models/" + modeldir + "/" + modeldir + "_Unetlin.pt"
    newmodel = dataset + "-models/" + modeldir + "/" + modeldir + "_statedictlin.pt"
    print(model)
    print(newmodel)
else:
    model = dataset + "-models/" + modeldir + "/" + modeldir + "_Unet.pt"
    newmodel = dataset + "-models/" + modeldir + "/" + modeldir + "_statedict.pt"
    print(model)
    print(newmodel)

unet_load,history_load = torch.load(model)

if (adapters): 
    layer_names = [name for name, module in unet_load.named_modules() if name.startswith("adapter")]
    layer_modules = [module for name, module in unet_load.named_modules() if name.startswith("adapter")]
    new_state_dict = {}
    for iname,name in enumerate(layer_names):
        new_state_dict[name] = layer_modules[iname].state_dict()
    torch.save(new_state_dict,newmodel)

if (bnorm and adapters): 
    bn_layer_names = [name for name, module in unet_load.named_modules() if name.endswith("_batchnorm")]
    bn_layer_modules = [module for name, module in unet_load.named_modules() if name.endswith("_batchnorm")]
    new_state_dict_bn = {}
    for iname,name in enumerate(bn_layer_names):
        new_state_dict_bn[name] = bn_layer_modules[iname].state_dict()

    layer_names = [name for name, module in unet_load.named_modules() if name.startswith("adapter")]
    layer_modules = [module for name, module in unet_load.named_modules() if name.startswith("adapter")]
    new_state_dict = {}
    for iname,name in enumerate(layer_names):
        new_state_dict[name] = layer_modules[iname].state_dict()
    torch.save(new_state_dict_bn | new_state_dict,newmodel)

if (not adapters): torch.save(unet_load.state_dict(),newmodel)

print(modeldir)
print(os.stat(newmodel).st_size/1024/1024,"MBytes")