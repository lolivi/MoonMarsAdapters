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

#------------------------------------------------
#--------------------BASELINE--------------------
#------------------------------------------------

#modelsdir
modelsdir = "synthetic-moon-models/"

encoder_names = ["resnet18","resnet34","resnet50","vgg11_bn","vgg13_bn","vgg16_bn","vgg19_bn","mobilenet_v2"]
params_train = []
params_perc = []
flops_arr = []
balaccs = [96.49,96.46,96.35,96.09,96.16,96.46,96.58,94.89]

for encoder_name in encoder_names:

    #baseline study
    baseline = True
    ftuneenc = False
    ftunedec = False
    ftunebnorm = False
    ftuneadapt = False

    unet_train = saveorload(modelsdir,1,0,0,0,0,0.5,encoder_name,"adam","BalancedCCE",baseline,ftuneenc,ftunedec,ftunebnorm,ftuneadapt)
    model_train(unet_train, encoder_name, baseline, ftuneenc, ftunedec, ftunebnorm, ftuneadapt)
    non_frozen_parameters = sum([len(torch.flatten(p)) for p in unet_train.parameters() if p.requires_grad])
    tot_parameters = sum([len(torch.flatten(p)) for p in unet_train.parameters()])
    params_train.append(non_frozen_parameters / 1e6)
    params_perc.append(non_frozen_parameters / tot_parameters * 100)

    #flop counter
    img_test = torch.rand([1,1,512,512])
    flops = FlopCountAnalysis(unet_train, img_test)
    nflops = flops.total() / 1e9
    flops_arr.append(nflops)

print("----------------------------------------------------")
print("Baseline study in synthetic-moon-models:")
print("Encoders: ",encoder_names)
print("Trainable Parameters (M): ",params_train)
print("Trainable Parameters (%s): " % ("%"),params_perc) 
print("FLOPs (G):",flops_arr)
print("----------------------------------------------------")