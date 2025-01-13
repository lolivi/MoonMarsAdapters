import model_library
from model_library import *

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
# GLOBAL VARIABLES

compopt = False # whether to compute mean, std devs and distributions
linopt = False # whether to linearize Unet with adapters
datasettype = "synthetic-moon" #"synthetic-moon", "real-moon","ai4mars","marsdataset3","ai4mars-inference"

if (datasettype == "synthetic-moon"):
    IMAGE_PATH = "synthetic-moon-dataset/images/render/"
    MASK_PATH = "synthetic-moon-dataset/images/ground/"

if (datasettype == "real-moon"):
    IMAGE_PATH = "synthetic-moon-dataset/real_moon_images/images/"
    MASK_PATH = "synthetic-moon-dataset/real_moon_images/masks/"

if (datasettype == "ai4mars"):
    IMAGE_PATH = "ai4mars-dataset-merged-0.1/msl/images/edr/"
    MASK_PATH_TRAIN = "ai4mars-dataset-merged-0.1/msl/labels/train/"
    MASK_PATH_TEST = "ai4mars-dataset-merged-0.1/msl/labels/test/masked-gold-min3-100agree/"
    MASK_ROVER = "ai4mars-dataset-merged-0.1/msl/images/mxy/"
    RANGE_30M = "ai4mars-dataset-merged-0.1/msl/images/rng-30m/"

if (datasettype == "ai4mars-inference"):
    IMAGE_PATH = "ai4mars-dataset-merged-0.1/msl/images/edr/"
    MASK_PATH_TRAIN = "ai4mars-dataset-merged-0.1/msl/labels/train/"
    MASK_PATH_TEST = "ai4mars-dataset-merged-0.1/msl/labels/test/masked-gold-min3-100agree/"
    MASK_ROVER = "ai4mars-dataset-merged-0.1/msl/images/mxy/"
    RANGE_30M = "ai4mars-dataset-merged-0.1/msl/images/rng-30m/"

if (datasettype == "marsdataset3"):
    IMAGE_PATH = "MarsDataset-v3/images/"
    MASK_PATH = "MarsDataset-v3/annotations/"
    SKY_PATH = "MarsDataset-v3/sky-annotations/"

#directories for plots and models
figsdir = datasettype + "-results/"
modelsdir = datasettype + "-models/"
if (datasettype == "ai4mars-inference"): modelsdir = "marsdataset3-models/"

# Check whether the specified path exists or not
if (not os.path.exists(figsdir)): os.makedirs(figsdir)
if (not os.path.exists(modelsdir)): os.makedirs(modelsdir)

#check if cuda is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_id = torch.cuda.current_device()
print("Test on Device: ", device)
if (device != "cpu"):
    print("Device_id: ", device_id)
    print("GPU: ",torch.cuda.get_device_name(device_id))

# B/W transformation
black_white = True #transforms it into grayscale
if (black_white): color_channels = 1
else: color_channels = 3

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------

def test_network(encoder_name,aug_GaussianBlur,aug_ColorJitter,aug_HorizontalFlip,aug_Rotate,aug_RandomCrop,
                  opt,loss_function,baseline,ftuneenc,ftunedec,ftunebnorm,ftuneadapt,resfact,nadapters = 'all',method='wnorm_nparams'):

    removed_images_smo = ['0510','0598','1343','1415','1429','1454','1596','1635','1772','2243','2682','2693','2581','2989','3416','3430','3617',
                    '3811','3924','3950','3984','4127','4679','4767','5151','5492','5495','5857','6072','6331','7281','7912','8837','8845',
                    '8926','9524','9689','0146','0320','0362','0506','0918','0978','1347','2326','3912','4031','4343','4862','5971','6527',
                    '6854','8050','8068','9235', #test
                    '0118','0603','0774','0828','1694','1701','1763','1944','2281','2845','3100','3218','3414','3743','3994','4463','4718',
                    '4818','5342','5344','5443','5448','5637','5682','5687','6019','6257','6280','6387','6565','6632','6828','7116','7215',
                    '7855','7889','8072','8317','8455','8720','9085','9173','9189','9531','0009','0290','1828','3072','8674','8710','9335', #validation 
                    '0018','0151','0260','0319','0338','0346','0374','0383','0501','0537','0546','0617','0624','0642','0665','0718','0760',
                    '0810','0827','0899','0967','0970','1054','1091','1100','1173','1199','1202','1224','1254','1324','1353','1387','1556',
                    '1604','1636','1660','1718','1748','1774','1781','1824','1833','1845','1863','1877','1888','1936','2067','2081','2083',
                    '2129','2270','2310','2321','2345','2396','2459','2461','2530','2640','2680','2719','2769','2903','2965','3005','3057',
                    '3077','3119','3123','3165','3186','3281','3303','3326','3364','3379','3393','3396','3421','3428','3429','3440','3486',
                    '3549','3713','3797','3818','3904','3920','4101','4111','4254','4288','4317','4428','4480','4486','4588','4665','4677',
                    '4826','4831','4851','4889','4891','5012','5073','5111','5131','5201','5206','5221','5255','5277','5305','5313','5315',
                    '5358','5383','5389','5392','5463','5508','5529','5544','5571','5628','5653','5673','5693','5703','5728','5771','5803',
                    '5819','5824','5846','5898','5931','6001','6029','6042','6083','6090','6152','6155','6164','6197','6211','6245','6278',
                    '6287','6293','6315','6336','6390','6413','6431','6432','6510','6564','6619','6656','6722','6891','6952','6980','6985',
                    '6997','7014','7059','7063','7130','7154','7193','7203','7251','7257','7318','7341','7345','7401','7440','7534','7535',
                    '7549','7596','7628','7647','7707','7708','7755','7803','7845','7861','7887','7913','7984','7987','8034','8067','8089',
                    '8094','8102','8122','8138','8176','8231','8232','8236','8240','8256','8295','8378','8423','8479','8599','8640','8739',
                    '8745','8758','8898','8938','9016','9034','9064','9124','9127','9128','9207','9274','9356','9408','9439','9480','9502',
                    '9539','9641','9672','9707','9730','9742','0304','0555','0750','0772','0820','2005','2103','2093','4189','4215','5086',
                    '5542'] #training
    
    removed_images_rmo = ['TCAM21']

    #marsdataset
    removed_images_rma1 = []

    #ai4mars
    removed_images_rma2 = []

    #redundant training combinations
    if (baseline and (ftuneenc or ftunedec or ftunebnorm or ftuneadapt)): return
    if (ftuneenc and ftunedec and ftunebnorm): return

    #choosing removed images
    if (datasettype == "synthetic-moon"): removed_images = removed_images_smo
    if (datasettype == "real-moon"): removed_images = removed_images_rmo
    if (datasettype == "marsdataset3"): removed_images = removed_images_rma1
    if (datasettype == "ai4mars" or datasettype == "ai4mars-inference"): removed_images = removed_images_rma2
    
    #dataframe
    if (datasettype == "synthetic-moon" or datasettype == "real-moon"):
        df = create_df(removed_images, True, IMAGE_PATH, datasettype)
        df_removed = create_df(removed_images, False, IMAGE_PATH, datasettype)
        print('Total Images: ', len(df))
    
    if (datasettype == "ai4mars" or datasettype == "ai4mars-inference"):
        df_trainval = create_df(removed_images, True, MASK_PATH_TRAIN, "ai4mars")
        df_test = create_df(removed_images, True, MASK_PATH_TEST, "ai4mars")
        df_removed = create_df(removed_images, False, MASK_PATH_TRAIN, "ai4mars") + create_df(removed_images, False, MASK_PATH_TEST, "ai4mars")
        print('Total Images: ', len(df_trainval) + len(df_test))

    if (datasettype == "marsdataset3"):
        df_train = create_df(removed_images, True, IMAGE_PATH + 'train/', datasettype)
        df_val = create_df(removed_images, True, IMAGE_PATH + 'val/', datasettype)
        df_test = create_df(removed_images, True, IMAGE_PATH + 'test/', datasettype)
        df_removed = create_df(removed_images, False, IMAGE_PATH + 'train/', datasettype)  + create_df(removed_images, False, IMAGE_PATH + 'val/', datasettype) + create_df(removed_images, False, IMAGE_PATH + 'test/', datasettype) 
        print('Total Images: ', len(df_train) + len(df_val) + len(df_test))

    print('Removed Images: ', len(df_removed))

    #plotting example
    # if (datasettype == "synthetic-moon" or datasettype == "real-moon"): 
    #     plotinput(1,df,IMAGE_PATH,MASK_PATH,figsdir,datasettype)

    # if (datasettype == "ai4mars"):
    #     plotinput(1,df_test,IMAGE_PATH,MASK_PATH_TEST,figsdir,datasettype,test=True)

    # if (datasettype == "marsdataset3"):
    #     plotinput(7,df_train,IMAGE_PATH + 'train/',MASK_PATH + 'train/',figsdir,datasettype)

    #splitting in training, validation and test
    if (datasettype == "synthetic-moon" or datasettype == "real-moon"): 
        X_train, X_val, X_test = datasplitter(datasettype, df_tot = df)

    if (datasettype == "ai4mars" or datasettype == "ai4mars-inference"):
        X_train, X_val, X_test = datasplitter(datasettype, df_train = df_trainval, df_test = df_test)

    if (datasettype == "marsdataset3"):
        X_train, X_val, X_test = datasplitter(datasettype, df_train = df_train, df_val = df_val, df_test = df_test)

    X_removed = df_removed['id'].values

    #mean and standard deviations
    mean, std = getmeanstd(X_train, IMAGE_PATH, datasettype, comp = compopt)

    if (datasettype == "synthetic-moon" or datasettype == "real-moon"): #480x704
        npix_x_val, npix_y_val = 480,704

    if (datasettype == "ai4mars"): #1024x1024
        npix_x_val, npix_y_val = 1024,1024

    if (datasettype == "ai4mars-inference"): #1024x1024
        npix_x_val, npix_y_val = 512,512

    if (datasettype == "marsdataset3"): #512x512
        npix_x_val, npix_y_val = 512,512

    if ((datasettype == "synthetic-moon" or datasettype == "real-moon") and resfact > 1): npix_x_val, npix_y_val = 512,768
    npix_x_val, npix_y_val = int(npix_x_val/resfact), int(npix_y_val/resfact) 

    #no augmentation and mean and std devs are from training as we shouldn't know the distribution of test set in real case scenario
    t_train = A.Compose([A.Resize(npix_x_val,npix_y_val,always_apply=True,p=1)])
    t_val = A.Compose([A.Resize(npix_x_val,npix_y_val,always_apply=True,p=1)])
    t_test = A.Compose([A.Resize(npix_x_val,npix_y_val,always_apply=True,p=1)])

    #datasets
    if (datasettype == "synthetic-moon"):
        train_set = FakeMoonDataset(IMAGE_PATH, MASK_PATH, X_train, mean, std, t_train)
        val_set = FakeMoonDataset(IMAGE_PATH, MASK_PATH, X_val, mean, std, t_val)
        test_set = FakeMoonDataset(IMAGE_PATH, MASK_PATH, X_test, mean, std, t_test)
        removed_set = FakeMoonDataset(IMAGE_PATH, MASK_PATH, X_removed, mean, std, t_test)

    if (datasettype == "real-moon"):
        train_set = RealMoonDataset(IMAGE_PATH, MASK_PATH, X_train, mean, std, t_train)
        val_set = RealMoonDataset(IMAGE_PATH, MASK_PATH, X_val, mean, std, t_val)
        test_set = RealMoonDataset(IMAGE_PATH, MASK_PATH, X_test, mean, std, t_test)
        removed_set = RealMoonDataset(IMAGE_PATH, MASK_PATH, X_removed, mean, std, t_test)

    if (datasettype == "ai4mars" or datasettype == "ai4mars-inference"):
        train_set = AI4MarsDataset(IMAGE_PATH, MASK_PATH_TRAIN, RANGE_30M, X_train, mean, std, t_train, test = False)
        val_set = AI4MarsDataset(IMAGE_PATH, MASK_PATH_TRAIN, RANGE_30M, X_val, mean, std, t_val, test = False)
        test_set = AI4MarsDataset(IMAGE_PATH, MASK_PATH_TEST, RANGE_30M, X_test, mean, std, t_test, test = True)
        removed_set = []

    if (datasettype == "marsdataset3"):
        train_set = MarsDatasetv3(IMAGE_PATH + 'train/', MASK_PATH + 'train/', SKY_PATH + 'train/', X_train, mean, std, t_train)
        val_set = MarsDatasetv3(IMAGE_PATH + 'val/', MASK_PATH + 'val/', SKY_PATH + 'val/', X_val, mean, std, t_val)
        test_set = MarsDatasetv3(IMAGE_PATH + 'test/', MASK_PATH + 'test/', SKY_PATH + 'test/', X_test, mean, std, t_test)
        removed_set = []

    # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------

    #dataloader
    if (datasettype == "synthetic-moon"):
        batch_size_train = 1
        batch_size_val = 1

    if (datasettype == "real-moon"):
        batch_size_train = 1
        batch_size_val = 1

    if (datasettype == "ai4mars" or datasettype == "ai4mars-inference"):
        batch_size_train = 1
        batch_size_val = 1

    if (datasettype == "marsdataset3"):
        batch_size_train = 1
        batch_size_val = 1

    set_seed(42,device)
    train_loader = DataLoader(train_set, batch_size=batch_size_train, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size_val, shuffle=False)  
    test_loader = DataLoader(test_set, batch_size=batch_size_val, shuffle=False)  
    if (removed_set): removed_loader = DataLoader(removed_set, batch_size=batch_size_val, shuffle=False) 
    else: removed_loader = []  

    # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------

    #frequency distribution
    train_count, val_count, test_count = getdistcts(train_set,val_set,test_set,datasettype,comp = compopt)
    class_weights = torch.tensor([round(1./train_count[0]),round(1./train_count[1]),round(1./train_count[2])],device = device, dtype=torch.float)
    plot_freq(train_count, val_count, test_count, figsdir) #plot frequency distribution

    # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------

    #metrics
    metric_names = ["Accuracy","Balanced Accuracy","Jaccard Score"]

    # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------

    #no adapters...
    set_seed(42,device)

    # loading model
    run_name = run_builder(aug_GaussianBlur,aug_ColorJitter,aug_HorizontalFlip,aug_Rotate,aug_RandomCrop,
                           encoder_name,opt,loss_function,
                           baseline,ftuneenc,ftunedec,ftunebnorm,ftuneadapt)
    if (not os.path.exists(modelsdir + run_name + "/")): 
        print(modelsdir + run_name + "/" + " does not exist ...")
        return
    if (not os.path.exists(figsdir + run_name + "/")): os.makedirs(figsdir + run_name + "/")
    unet_load, history_load = torch.load(modelsdir + run_name + "/" + run_name + "_Unet.pt",map_location=device)
    if not(baseline or ftuneenc or ftunedec or ftunebnorm or ftuneadapt): 
        unet_load, history_load = torch.load("synthetic-moon-models/" + run_name + "/" + run_name + "_Unet.pt",map_location=device)

    if (linopt and ftuneadapt): 
        unet_load = compress_weights(unet_load, encoder_name)
        torch.save((unet_load,history_load), modelsdir + run_name + "/" + run_name + "_Unetlin.pt")
        unet_load.to(device = device)
        print("Compressed model produced")

    if (nadapters == 'tradeoff' and ftuneadapt):
        print('Finding best tradeoff to eliminate adapters...')
        nadapters = adaptertradeoff(copy.deepcopy(unet_load), val_loader, encoder_name, method, metric_names, 0.005, device)
        print('Removing %i adapters...' % nadapters)

    if (nadapters != 'all' and ftuneadapt):
        unet_load,wnorm,nparams,size,isorted = remove_adapters(unet_load, nadapters, encoder_name, method)
        used_adapters = [isortadapt + 1 for isortadapt in isorted][nadapters:]
        used_adapters.sort()
        print("Adapters used:", used_adapters)

    #loss function
    if loss_function == "BalancedCCE":
        criterion = ["BalancedCCE",nn.CrossEntropyLoss(weight = class_weights)]
    elif loss_function == "Jaccard":
        criterion = ["Jaccard",smp.losses.JaccardLoss(mode = 'multiclass',classes = [0,1,2])]
    elif loss_function == "Dice":
        criterion = ["Dice",smp.losses.DiceLoss(mode = 'multiclass',classes = [0,1,2])]

    #plotting results
    plot_loss(history_load, criterion[0], run_name + "/", figsdir)
    plot_score(history_load, metric_names, run_name + "/", figsdir)
    plot_lrs(history_load, run_name + "/",figsdir)

    dataloaders = [val_loader,test_loader,train_loader,removed_loader]
    datanames = [X_val,X_test,X_train,X_removed]
    datatests = [val_set,test_set,train_set,removed_set]
    datadirs = ["Validation","Test","Training","Removed"]
    plotopt = [False,False,False,False]

    dataloaders = [val_loader]
    datanames = [X_val]
    datatests = [val_set]
    datadirs = ["Validation"]
    plotopt = [True]

    for idata,dataloader in enumerate(dataloaders):
        
        #plots in each model
        figsrun = figsdir + run_name + "/"
        if not(baseline or ftuneenc or ftunedec or ftunebnorm or ftuneadapt): figsrun = figsdir + run_name + "_baseline/"
        if not os.path.exists(figsrun): os.mkdir(figsrun)
        dataset = datatests[idata]
        if (not dataset): continue
        scores, conf_matrix = evaluate(unet_load,dataloader,metric_names,device)

        plot_conf_matrix(conf_matrix,figsrun)
        acc = scores[0]
        bal_acc = scores[1]
        jacc_score = scores[2]

        print("Accuracy: {:.10f}".format(np.mean(acc)))
        print("Balanced Accuracy: {:.10f}".format(np.mean(bal_acc)))
        print("Jaccard Score: {:.10f}".format(np.mean(jacc_score)))

        mean_scores = [[np.mean(acc),np.mean(bal_acc),np.mean(jacc_score)]]
        column_names = copy.deepcopy(metric_names)
        resultsrun = figsrun + datadirs[idata] + "/"
        if not os.path.exists(resultsrun): os.mkdir(resultsrun)
        outfile = resultsrun + "metrics%i.txt" % resfact
        if (linopt): outfile = resultsrun + "metricslin%i.txt" % resfact
        if (nadapters != 'all' and ftuneadapt): 
            outfile = resultsrun + ("metricsadapt%i" % nadapters) + "_" + method + ".txt"
            mean_scores[0].extend([wnorm,nparams,size])
            column_names.extend(["Weight_Norm","N_Parameters","Adapter_Size"])
        column_names = [c.replace(' ','_') for c in column_names]
        df_results = pd.DataFrame(data = mean_scores, columns = column_names)
        df_results.to_csv(outfile, index = False, sep = " ")

        if not plotopt[idata]: continue

        for iplot in range(len(dataloader)):
            
            dataset_im = dataset[iplot]
            image,mask = dataset_im[0],dataset_im[1]
            prediction = predicted_mask(unet_load,image.view(1,1,npix_x_val,npix_y_val),device)
            if (not os.path.exists(figsrun + datadirs[idata] + "/")): os.makedirs(figsrun + datadirs[idata] + "/")
            plot_prediction(image,mask,prediction,datanames[idata][iplot],bal_acc[iplot],iplot,npix_x_val,npix_y_val,figsrun + datadirs[idata] + "/")

        computememory()


#--------------------------------------------
#--------------------MAIN--------------------
#---------------------------------------------

# resfacts = [1,2,4]
# encoders = ["resnet18","vgg19_bn"]
# for encoder in encoders:
#     for resfact in resfacts:
#         test_network(encoder,0,0,0,0,0.5,"adam","BalancedCCE",False,False,False,False,True,resfact)
        
# encoders = ["resnet18","vgg19_bn"]
# ftunes = [[False,False,False,False,False],
#           [True,False,False,False,False],
#           [False,True,True,False,False],
#           [False,True,False,False,False],
#           [False,False,True,False,False],
#           [False,False,False,True,False],
#           [False,False,False,False,True]]

#test_network("vgg19_bn",0,0,0,0,0.5,"adam","BalancedCCE",False,False,False,False,True,1)

'''
e = "vgg19_bn"
if (e == "resnet18"): nadapters = 28
if (e == "vgg19_bn"): nadapters = 29
methods = ["wnorm_nparams","wnorm","nparams","forward","backward"]
for method in methods:
    for nadapter in range(nadapters):
        test_network(e,0,0,0,0,0.5,"adam","BalancedCCE",False,False,False,False,True,1,nadapter,method)
# '''

'''
encoders = ["vgg19_bn"]  
for encoder in encoders:
    test_network(encoder,0,0,0,0,0.5,"adam","BalancedCCE",False,False,False,False,True,1,'tradeoff','wnorm_nparams')
'''
    
# for nadapter in range(29):
#     test_network("vgg19_bn",0,0,0,0,0.5,"adam","BalancedCCE",False,False,False,False,True,1,nadapter,'wnorm_nparams')

encoders = ["resnet18","vgg19_bn"]  
for encoder in encoders:
    #test_network(encoder,0,0,0,0,0.5,"adam","BalancedCCE",False,False,False,False,True,1)
    test_network(encoder,0,0,0,0,0.5,"adam","BalancedCCE",True,False,False,False,False,1)