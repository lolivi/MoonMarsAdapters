import model_library
from model_library import *

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
# GLOBAL VARIABLES

#deciding whether to log or not variables to wandb
wandbopt = True
compopt = False # whether to compute mean, std devs and distributions
datasettype = "synthetic-moon" #"synthetic-moon", "real-moon","ai4mars","marsdataset3"

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

if (datasettype == "marsdataset3"):
    IMAGE_PATH = "MarsDataset-v3/images/"
    MASK_PATH = "MarsDataset-v3/annotations/"
    SKY_PATH = "MarsDataset-v3/sky-annotations/"

#directories for plots and models
figsdir = datasettype + "-figs/"
modelsdir = datasettype + "-models/"

# Check whether the specified path exists or not
if (not os.path.exists(figsdir)): os.makedirs(figsdir)
if (not os.path.exists(modelsdir)): os.makedirs(modelsdir)

#check if cuda is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_id = torch.cuda.current_device()
print("Training on Device: ", device)
print("Device_id: ", device_id)
print("GPU: ",torch.cuda.get_device_name(device_id))

# B/W transformation
black_white = True #transforms it into grayscale
if (black_white): color_channels = 1
else: color_channels = 3

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------

def train_network(encoder_name,aug_GaussianBlur,aug_ColorJitter,aug_HorizontalFlip,aug_Rotate,aug_RandomCrop,
                  opt,loss_function,baseline,ftuneenc,ftunedec,ftunebnorm,ftuneadapt,run,wandbopt = False):

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
    if not(baseline or ftuneenc or ftunedec or ftunebnorm or ftuneadapt): return
    if (ftuneenc and ftunedec and ftunebnorm): return

    #choosing removed images
    if (datasettype == "synthetic-moon"): removed_images = removed_images_smo
    if (datasettype == "real-moon"): removed_images = removed_images_rmo
    if (datasettype == "marsdataset3"): removed_images = removed_images_rma1
    if (datasettype == "ai4mars"): removed_images = removed_images_rma2
    
    #dataframe
    if (datasettype == "synthetic-moon" or datasettype == "real-moon"):
        df = create_df(removed_images, True, IMAGE_PATH, datasettype)
        df_removed = create_df(removed_images, False, IMAGE_PATH, datasettype)
        print('Total Images: ', len(df))
    
    if (datasettype == "ai4mars"):
        df_trainval = create_df(removed_images, True, MASK_PATH_TRAIN, datasettype)
        df_test = create_df(removed_images, True, MASK_PATH_TEST, datasettype)
        df_removed = create_df(removed_images, False, MASK_PATH_TRAIN, datasettype) + create_df(removed_images, False, MASK_PATH_TEST, datasettype)
        print('Total Images: ', len(df_trainval) + len(df_test))

    if (datasettype == "marsdataset3"):
        df_train = create_df(removed_images, True, IMAGE_PATH + 'train/', datasettype)
        df_val = create_df(removed_images, True, IMAGE_PATH + 'val/', datasettype)
        df_test = create_df(removed_images, True, IMAGE_PATH + 'test/', datasettype)
        df_removed = create_df(removed_images, False, IMAGE_PATH + 'train/', datasettype)  + create_df(removed_images, False, IMAGE_PATH + 'val/', datasettype) + create_df(removed_images, False, IMAGE_PATH + 'test/', datasettype) 
        print('Total Images: ', len(df_train) + len(df_val) + len(df_test))

    print('Removed Images: ', len(df_removed))

    #plotting example
    if (datasettype == "synthetic-moon" or datasettype == "real-moon"): 
        plotinput(1,df,IMAGE_PATH,MASK_PATH,figsdir,datasettype)

    if (datasettype == "ai4mars"):
        plotinput(1,df_test,IMAGE_PATH,MASK_PATH_TEST,figsdir,datasettype,test=True)

    if (datasettype == "marsdataset3"):
        plotinput(7,df_train,IMAGE_PATH + 'train/',MASK_PATH + 'train/',figsdir,datasettype)

    #splitting in training, validation and test
    if (datasettype == "synthetic-moon" or datasettype == "real-moon"): 
        X_train, X_val, X_test = datasplitter(datasettype, df_tot = df)

    if (datasettype == "ai4mars"):
        X_train, X_val, X_test = datasplitter(datasettype, df_train = df_trainval, df_test = df_test)

    if (datasettype == "marsdataset3"):
        X_train, X_val, X_test = datasplitter(datasettype, df_train = df_train, df_val = df_val, df_test = df_test)

    #mean and standard deviations
    mean, std = getmeanstd(X_train, IMAGE_PATH, datasettype, comp = compopt)

    if (datasettype == "synthetic-moon" or datasettype == "real-moon"): #480x704
        npix_x_train, npix_y_train = 224,352
        npix_x_val, npix_y_val = 480,704
        npix_x_crop, npix_y_crop = 224,352

    if (datasettype == "ai4mars"): #1024x1024
        npix_x_train, npix_y_train = 256,256
        npix_x_val, npix_y_val = 512,512
        npix_x_crop, npix_y_crop = 512,512

    if (datasettype == "marsdataset3"): #512x512
        npix_x_train, npix_y_train = 256,256
        npix_x_val, npix_y_val = 512,512
        npix_x_crop, npix_y_crop = 256,256

    t_train = A.Compose([A.GaussianBlur(p=aug_GaussianBlur),
                        A.ColorJitter(p=aug_ColorJitter),
                        A.HorizontalFlip(p=aug_HorizontalFlip), 
                        A.Rotate(p=aug_Rotate),
                        A.RandomCrop(npix_x_crop,npix_y_crop,p=aug_RandomCrop), #crop of half the image with same ratio,
                        A.Resize(npix_x_train,npix_y_train,always_apply=True,p=1)
                        ])
    # t_train=A.Compose([])

    #no augmentation and mean and std devs are from training as we shouldn't know the distribution of test set in real case scenario
    t_val = A.Compose([A.Resize(npix_x_val,npix_y_val,always_apply=True,p=1)])
    t_test = A.Compose([A.Resize(npix_x_val,npix_y_val,always_apply=True,p=1)])

    #datasets
    if (datasettype == "synthetic-moon"):
        train_set = FakeMoonDataset(IMAGE_PATH, MASK_PATH, X_train, mean, std, t_train)
        val_set = FakeMoonDataset(IMAGE_PATH, MASK_PATH, X_val, mean, std, t_val)
        test_set = FakeMoonDataset(IMAGE_PATH, MASK_PATH, X_test, mean, std, t_test)

    if (datasettype == "real-moon"):
        train_set = RealMoonDataset(IMAGE_PATH, MASK_PATH, X_train, mean, std, t_train)
        val_set = RealMoonDataset(IMAGE_PATH, MASK_PATH, X_val, mean, std, t_val)
        test_set = RealMoonDataset(IMAGE_PATH, MASK_PATH, X_test, mean, std, t_test)

    if (datasettype == "ai4mars"):
        train_set = AI4MarsDataset(IMAGE_PATH, MASK_PATH_TRAIN, RANGE_30M, X_train, mean, std, t_train, test = False)
        val_set = AI4MarsDataset(IMAGE_PATH, MASK_PATH_TRAIN, RANGE_30M, X_val, mean, std, t_val, test = False)
        test_set = AI4MarsDataset(IMAGE_PATH, MASK_PATH_TEST, RANGE_30M, X_test, mean, std, t_test, test = True)

    if (datasettype == "marsdataset3"):
        train_set = MarsDatasetv3(IMAGE_PATH + 'train/', MASK_PATH + 'train/', SKY_PATH + 'train/', X_train, mean, std, t_train)
        val_set = MarsDatasetv3(IMAGE_PATH + 'val/', MASK_PATH + 'val/', SKY_PATH + 'val/', X_val, mean, std, t_val)
        test_set = MarsDatasetv3(IMAGE_PATH + 'test/', MASK_PATH + 'test/', SKY_PATH + 'test/', X_test, mean, std, t_test)

    # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------

    #plot_image_mask(train_set, 8, figsdir) 

    #dataloader
    if (datasettype == "synthetic-moon"):
        batch_size_train = 32
        batch_size_val = 16

    if (datasettype == "real-moon"):
        batch_size_train = 5
        batch_size_val = 1

    if (datasettype == "ai4mars"):
        batch_size_train = 16
        batch_size_val = 16

    if (datasettype == "marsdataset3"):
        batch_size_train = 14
        batch_size_val = 16

    set_seed(42,device)
    train_loader = DataLoader(train_set, batch_size=batch_size_train, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size_val, shuffle=False)   

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

    #getting unet
    unet_train = saveorload(modelsdir,color_channels,
                            aug_GaussianBlur, aug_ColorJitter, aug_HorizontalFlip, aug_Rotate, aug_RandomCrop, 
                            encoder_name, opt, loss_function,
                            baseline,ftuneenc,ftunedec,ftunebnorm,ftuneadapt)

    # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # GridSearch

    if loss_function == "BalancedCCE":
        criterion = ["BalancedCCE",nn.CrossEntropyLoss(weight = class_weights)]
    elif loss_function == "Jaccard":
        criterion = ["Jaccard",smp.losses.JaccardLoss(mode = 'multiclass',classes = [0,1,2])]
    elif loss_function == "Dice":
        criterion = ["Dice",smp.losses.DiceLoss(mode = 'multiclass',classes = [0,1,2])]

    weigth_decay = 0

    # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------

    #training loop
    model_train(unet_train, encoder_name, baseline, ftuneenc, ftunedec, ftunebnorm, ftuneadapt)
    non_frozen_parameters = [p for p in unet_train.parameters() if p.requires_grad]
    print("Trainable Parameters: ", len(non_frozen_parameters))

    if (opt == "adam"):
        optimizer = torch.optim.AdamW(non_frozen_parameters, lr=1e-3, weight_decay=weigth_decay)

    if (opt == "sgdm"):
        optimizer = torch.optim.SGD(non_frozen_parameters, lr=1e-1, momentum = 0.9, weight_decay=weigth_decay)

    #scheduler
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, threshold=0, threshold_mode='abs')
    #sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=30, threshold=0, threshold_mode='abs')

    #training
    history = fit(unet_train, train_loader, val_loader, criterion[1], optimizer, sched, metric_names, 
                  baseline, ftuneenc, ftunedec, ftunebnorm, ftuneadapt, encoder_name, 
                  wandbopt, run, device)

    #saving model
    run_name = run_builder(aug_GaussianBlur,aug_ColorJitter,aug_HorizontalFlip,aug_Rotate,aug_RandomCrop,
                           encoder_name,opt,loss_function,
                           baseline,ftuneenc,ftunedec,ftunebnorm,ftuneadapt)

    if (not os.path.exists(modelsdir + run_name + "/")): os.makedirs(modelsdir + run_name + "/")
    if (not os.path.exists(figsdir + run_name + "/")): os.makedirs(figsdir + run_name + "/")
    torch.save((unet_train,history), modelsdir + run_name + "/" + run_name + "_Unet.pt")

    #plotting results
    plot_loss(history,criterion[0], run_name + "/", figsdir)
    plot_score(history,metric_names, run_name + "/", figsdir)
    plot_lrs(history, run_name + "/",figsdir)

    # saving the hyperparameters
    hypers = [get_lr(optimizer), criterion[0], opt]
    header = "InitialLR Loss Optimizer"
    header = header.split()
    df = pd.DataFrame(data = [hypers], columns = header)
    df.to_csv(modelsdir + run_name + "/" + run_name +"_hypers.txt", index = False, sep = " ", mode = "w", header = True)

    gc.collect()
    # gc.set_debug(gc.DEBUG_LEAK)
    # print(gc.get_objects)


    computememory()

# end function

#--------------------------------------------
#--------------------MAIN--------------------
#---------------------------------------------

if (wandbopt):

    # start a new wandb run to track this script
    run = wandb.init()

    #wandb sweep name
    run_name = run_builder(wandb.config.aug_GaussianBlur,wandb.config.aug_ColorJitter,wandb.config.aug_HorizontalFlip,wandb.config.aug_Rotate,wandb.config.aug_RandomCrop,
                           wandb.config.encoder,wandb.config.optimizer,wandb.config.loss_function,
                           wandb.config.baseline,wandb.config.ftuneenc,wandb.config.ftunedec,wandb.config.ftunebnorm,wandb.config.ftuneadapt)
    run.name = run_name

    #saving code in wandb manually
    run.log_code(".")

    #calling function
    train_network(wandb.config.encoder,wandb.config.aug_GaussianBlur,wandb.config.aug_ColorJitter,wandb.config.aug_HorizontalFlip,wandb.config.aug_Rotate,
                wandb.config.aug_RandomCrop,wandb.config.optimizer,wandb.config.loss_function,
                wandb.config.baseline,wandb.config.ftuneenc,wandb.config.ftunedec,wandb.config.ftunebnorm,wandb.config.ftuneadapt,
                run,wandbopt)

    run.finish()

else:

    run_name = "test"

    # calling function
    train_network("vgg19_bn",0,0,0,0,
                0.5,"adam","BalancedCCE",
                False,False,False,False,True,
                run_name,wandbopt)


