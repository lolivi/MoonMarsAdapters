import model_library
from model_library import *

modelsdirs = ["real-moon-models","marsdataset3-models"]
encoders = ["vgg19_bn","resnet18"]
means = [[0.3748141096978818, 0.35596510164602224, 0.2955574219124973],[0.6096051074840404, 0.5055833881828149, 0.35589452779742]]
std_devs = [[0.218745767389803, 0.2105994343046987, 0.18040353055725178],[0.14303784363117197, 0.12331537394970005, 0.0983118132001638]]
imagespath = ["synthetic-moon-dataset/real_moon_images/images/TCAM4.png","MarsDataset-v3/images/test/cr_079_0079MR0005910090103969E01_DXXX_raw.png"]
removetradeoff = [11,9]
baselines = [False,False,False,False]
encs = [False,False,False,True]
decs = [False,False,False,True]
bnorms = [False,False,False,False]
adapts = [False,True,True,False]
xlegends = ["Baseline","Adapter (Full)","Adapter (Ranked)","Full Finetuning"]
ylegends = ["RealMoon","MarsDataset"]
iremove = 2
t_resize = A.Compose([A.Resize(512,512,always_apply=True,p=1)])

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

# Create a 2x3 grid of subplots
fig, axs = plt.subplots(len(modelsdirs), len(baselines), figsize=(15, 8))

for imodel,modelsdir in enumerate(modelsdirs):
    for ibal in range(len(baselines)):

        encoder_name = encoders[imodel]
        baseline = baselines[ibal]
        ftuneenc = encs[ibal]
        ftunedec = decs[ibal]
        ftunebnorm = bnorms[ibal]
        ftuneadapt = adapts[ibal]

        run_name = run_builder(0,0,0,0,0.5,
                           encoder_name,"adam","BalancedCCE",
                           baseline,ftuneenc,ftunedec,ftunebnorm,ftuneadapt)
        
        if (ibal == 0): unet_load, history_load = torch.load("synthetic-moon-models/" + run_name + "/" + run_name + "_Unet.pt",map_location="cpu")
        else: unet_load, history_load = torch.load(modelsdir + "/" + run_name + "/" + run_name + "_Unet.pt",map_location="cpu")
        if (ibal == iremove): 
            if (imodel == 0): 
                '''
                unet_ranked = Unet_vgg19_ranked(1,3,True)
                unet_ranked.load_state_dict(unet_load.state_dict())
                unet_load = copy.deepcopy(unet_ranked)
                '''
                unet_load,wnorm,nparams,size,isorted = remove_adapters(unet_load,removetradeoff[imodel],encoder_name,"wnorm_nparams")
            if (imodel == 1): 
                '''
                unet_ranked = Unet_18_ranked(1,3,True)
                unet_ranked.load_state_dict(unet_load.state_dict())
                unet_load = copy.deepcopy(unet_ranked)
                '''
                unet_load,wnorm,nparams,size,isorted = remove_adapters(unet_load,removetradeoff[imodel],encoder_name,"wnorm_nparams")

        #lettura immagini
        img = cv2.imread(imagespath[imodel])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #from BGR to RGB
        
        #transformations
        aug = t_resize(image=img)
        img = Image.fromarray(aug['image'])
        
        #normalization
        t_bw = T.Compose([T.ToTensor(), T.Normalize(means[imodel], std_devs[imodel]), T.Grayscale()])
        image = t_bw(img)

        #prediction
        prediction = predicted_mask(unet_load,image.view(1,1,512,512),"cpu")

        axs[imodel][ibal].imshow(image.permute(1, 2, 0),cmap="gray")
        im = axs[imodel][ibal].imshow(prediction.cpu().view(512,512), alpha = 0.6, cmap=cm, norm=norm)
        axs[imodel][ibal].set_xticks([])
        axs[imodel][ibal].set_yticks([])

        if (ibal == 0): axs[imodel][ibal].set_ylabel(ylegends[imodel],fontsize = 14)
        if (imodel == 0): axs[imodel][ibal].set_title(xlegends[ibal],fontsize = 14)

        #plt.axis('off')

cbar_ax = fig.add_axes([0.32, 0.05, 0.4, 0.03])  # [left, bottom, width, height]
diff = norm_bins[1:] - norm_bins[:-1]
tickz = norm_bins[:-1] + diff / 2
cb = fig.colorbar(im, cax = cbar_ax, format=fmt, ticks=tickz, orientation='horizontal')
plt.subplots_adjust(wspace=0, hspace=0.1)
#plt.tight_layout()
plt.savefig("adapt-domains-equal.pdf", bbox_inches="tight")

sys.exit()

# Sample data
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)
y3 = np.tan(x)

# Create a 2x3 grid of subplots
fig, axs = plt.subplots(len(modelsdirs), len(baselines), figsize=(12, 8))

# Top row plots (Row 1)
axs[0, 0].plot(x, y1)
axs[0, 0].set_title("Sine Plot")
axs[0, 1].plot(x, y2)
axs[0, 1].set_title("Cosine Plot")
axs[0, 2].plot(x, y3)
axs[0, 2].set_title("Tangent Plot")

# Row title for top row
fig.suptitle('Title for Row 1', x=0.5, y=0.97, fontsize=14)

# Bottom row plots (Row 2)
axs[1, 0].plot(x, -y1)
axs[1, 1].plot(x, -y2)
axs[1, 2].plot(x, -y3)

# Row title for bottom row
fig.text(0.5, 0.52, 'Title for Row 2', ha='center', fontsize=14)

# Remove space between subplots
plt.subplots_adjust(wspace=0, hspace=0.3)

plt.show()