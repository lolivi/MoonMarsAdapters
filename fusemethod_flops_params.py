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

#modelsdir
modelsdir = "real-moon-models"     # marsdataset3-models,real-moon-models

#adapter method
baseline = False
ftuneenc = True
ftunedec = True
ftunebnorm = True
ftuneadapt = True


#fused methods
unet_resnet18 = saveorload(modelsdir,1,0,0,0,0,0.5,"resnet18","adam","BalancedCCE",baseline,ftuneenc,ftunedec,ftunebnorm,ftuneadapt)
model_train(unet_resnet18, "resnet18", False, True, True, True, True)
unet_resnet18_fuse = compress_weights(copy.deepcopy(unet_resnet18),"resnet18")

non_frozen_parameters_resnet_orig = sum([len(torch.flatten(p)) for p in unet_resnet18.parameters() if p.requires_grad])
non_frozen_parameters_resnet_fuse = sum([len(torch.flatten(p)) for p in unet_resnet18_fuse.parameters() if p.requires_grad])


unet_vgg19 = saveorload(modelsdir,1,0,0,0,0,0.5,"vgg19_bn","adam","BalancedCCE",baseline,ftuneenc,ftunedec,ftunebnorm,ftuneadapt)
model_train(unet_vgg19, "vgg19_bn", False, True, True, True, True)
unet_vgg19_fuse = compress_weights(copy.deepcopy(unet_vgg19),"vgg19_bn")

non_frozen_parameters_vgg19_orig = sum([len(torch.flatten(p)) for p in unet_vgg19.parameters() if p.requires_grad])
non_frozen_parameters_vgg19_fuse = sum([len(torch.flatten(p)) for p in unet_vgg19_fuse.parameters() if p.requires_grad])



params_train= [non_frozen_parameters_resnet_orig/1e6,non_frozen_parameters_resnet_fuse/1e6,non_frozen_parameters_vgg19_orig/1e6,non_frozen_parameters_vgg19_fuse/1e6] 



#flop counter
img_test = torch.ones([1,1,512,512])
flops_arr = []

#methods
methods = ["resnet18", "resnet18-fuse", "vgg19", "vgg19-fuse"]
unets = [unet_resnet18, unet_resnet18_fuse, unet_vgg19, unet_vgg19_fuse]
for imethod,method in enumerate(methods):
    flops = FlopCountAnalysis(copy.deepcopy(unets[imethod]), img_test)
    nflops = flops.total() / 1e9
    flops_arr.append(nflops)

print("----------------------------------------------------")
print("Adapter Fusion:")
print("Methods: ", methods)
print("Trainable Parameters [M]: ", params_train) 
print("FLOPs (G):", flops_arr)
print("----------------------------------------------------")