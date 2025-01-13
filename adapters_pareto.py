import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt

#list inputs
datasets = ["marsdataset3","real-moon"]
encoders = ["resnet18","vgg19_bn"]
removetradeoff = [9,11]
xlabels = ["Resnet18 - Adapter Cumulative Size [MBytes]","Vgg19BN - Adapter Cumulative Size [MBytes]"]
ylabels = ["Real Mars - Balanced Accuracy [%]","Real Moon - Balanced Accuracy [%]"]
'''
colors = ["green", "red", "cyan", "orange","blue"]
methods = ["forward","backward","wnorm","nparams","wnorm_nparams"]
labels = ["Forward","Backward", r"$Z \ = \ ||w_{f_n}||^2$", r"$Z \ = \ |w_{f_n}|$",r"$Z \ = \ \frac{||w_{f_n}||^2}{|w_{f_n}|}$"]
'''
colors = ["cyan", "orange","blue"]
methods = ["wnorm","nparams","wnorm_nparams"]
labels = [r"$Z \ = \ ||w_{f_n}||^2$", r"$Z \ = \ |w_{f_n}|$",r"$Z \ = \ \frac{||w_{f_n}||^2}{|w_{f_n}|}$"]
 
fig, ax = plt.subplots(1,2,figsize = (12,5),clear = True)
for ienc,encoder in enumerate(encoders):

    if encoder == "resnet18": nadapters = 27
    if encoder == "vgg19_bn": nadapters = 28

    #directories
    modelname = encoder + "_adam_BalancedCCE_RandomCrop_adapter"
    modeldir = datasets[ienc] + "-results/" + modelname + "/Test/"

    #loop on methods
    for imeth,method in enumerate(methods):
        
        balaccs = []
        sizes = []
        for iadapt in range(nadapters+1):
            if (iadapt == removetradeoff[ienc] and imeth == 0): besttradeoff = iadapt
            resultsadapt = pd.read_csv(modeldir + ("metricsadapt%i" % iadapt) + "_" + method + ".txt",delimiter=' ')
            balaccs.append(resultsadapt['Balanced_Accuracy'].values[0] * 100)
            sizes.append(resultsadapt['Adapter_Size'].values[0])

        #plotting every method
        ax[ienc].plot(sizes,balaccs,color = colors[imeth],label = labels[imeth],linestyle = '-',marker = 'o',markersize = 3)
        if (method == "wnorm_nparams"): ax[ienc].plot(sizes[besttradeoff],balaccs[besttradeoff],color = "blue",label = 'Best Trade-Off',linestyle = '-',marker = 'o',markersize = 10)
    
    ax[ienc].set_xlabel(xlabels[ienc])
    ax[ienc].set_ylabel(ylabels[ienc])
    ax[ienc].grid(True, which='major')
    ax[ienc].grid(True, which='minor', alpha=0.2)
    ax[ienc].set_axisbelow(True)
    ax[ienc].minorticks_on()

# Create a single legend for both subplots
handles, labels = ax[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=7, bbox_to_anchor=(0.5, 1.05))

# Adjust layout to make space for the legend
fig.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("adapt_pareto.pdf", bbox_inches='tight')

sys.exit()