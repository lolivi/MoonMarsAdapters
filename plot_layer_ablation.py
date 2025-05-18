import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


dataset = 'real-moon'    # real-moon, marsdataset3

dir = dataset+'-results/'


bal_accs = []
i_layer = []
for i in range(28):
    run_name = dir+"ablation_layer_%i/Test/metrics1.txt" % i
    p = pd.read_csv(run_name, delimiter=' ')
    bal_accs.append(p['Balanced_Accuracy'].values[0]*100)
    i_layer.append(i+1)


if dataset=='real-moon':
    df = pd.read_csv(dir+"vgg19_bn_adam_BalancedCCE_RandomCrop/Test/metrics1.txt", delim_whitespace=True)
    bal_acc_base = df.iloc[0,1]*100
    df = pd.read_csv(dir+"vgg19_bn_adam_BalancedCCE_RandomCrop_adapter/Test/metrics1.txt", delim_whitespace=True)
    bal_acc_adapt = df.iloc[0,1]*100

    print(bal_acc_adapt)
    plt.figure()
    plt.xlabel("Vgg19BN - Layer")
    plt.bar(i_layer, bal_accs, color = 'skyblue')
    plt.grid(True, which='major')
    plt.grid(True, which='minor', alpha=0.2)
    plt.axhline(bal_acc_base, color='black', linestyle='--', label="Baseline")
    plt.axhline(bal_acc_adapt, color='black', linestyle='--', label="Adapter")
    plt.ylim(75,95)
    plt.ylabel("Balanced Accuracy [%]")
    plt.title("Real Moon - Layer Ablation Study")
    plt.savefig("bal_accs_realmoon.pdf")


else:
    df = pd.read_csv(dir+"resnet18_adam_BalancedCCE_RandomCrop/Test/metrics1.txt", delim_whitespace=True)
    bal_acc_base = df.iloc[0,1]*100
    df = pd.read_csv(dir+"resnet18_adam_BalancedCCE_RandomCrop_adapter/Test/metrics1.txt", delim_whitespace=True)
    bal_acc_adapt = df.iloc[0,1]*100
    
    
    plt.figure()
    plt.xlabel("Resnet18 - Layer")
    plt.bar(i_layer, bal_accs, color = 'skyblue')
    plt.grid(True, which='major')
    plt.grid(True, which='minor', alpha=0.2)
    plt.axhline(bal_acc_base, color='black', linestyle='--', label="Baseline")
    plt.axhline(bal_acc_adapt, color='black', linestyle='--', label="Adapter")
    plt.ylim(40,100)
    plt.ylabel("Balanced Accuracy [%]")
    plt.title("MarsDatset - Layer Ablation Study")
    plt.savefig("bal_accs_marsdataset.pdf")



