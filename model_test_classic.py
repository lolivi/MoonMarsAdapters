import model_library_classic
from model_library_classic import *

datasettype = "marsdataset3" #"synthetic-moon", "real-moon","ai4mars","marsdataset3","ai4mars-inference"
run_name = "otsu"   # otsu, canny, hybrid

print(" ")
print("Dataset: " + datasettype)
print(" ")
print("Algorithm: " + run_name)
print(" ")
print(" ")


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
figsdir = datasettype + "-figs/"

# Check whether the specified path exists or not
if (not os.path.exists(figsdir)): os.makedirs(figsdir)

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


#splitting in training, validation and test
if (datasettype == "synthetic-moon" or datasettype == "real-moon"): 
    X_train, X_val, X_test = datasplitter(datasettype, df_tot = df)

if (datasettype == "ai4mars" or datasettype == "ai4mars-inference"):
    X_train, X_val, X_test = datasplitter(datasettype, df_train = df_trainval, df_test = df_test)

if (datasettype == "marsdataset3"):
    X_train, X_val, X_test = datasplitter(datasettype, df_train = df_train, df_val = df_val, df_test = df_test)

X_removed = df_removed['id'].values

if (datasettype == "synthetic-moon" or datasettype == "real-moon"): #480x704
    npix_x_val, npix_y_val = 480,704

if (datasettype == "ai4mars"): #1024x1024
    npix_x_val, npix_y_val = 1024,1024

if (datasettype == "ai4mars-inference"): #1024x1024
    npix_x_val, npix_y_val = 512,512

if (datasettype == "marsdataset3"): #512x512
    npix_x_val, npix_y_val = 512,512

#no augmentation and mean and std devs are from training as we shouldn't know the distribution of test set in real case scenario
t_train = A.Compose([A.Resize(npix_x_val,npix_y_val,always_apply=True,p=1)])
t_val = A.Compose([A.Resize(npix_x_val,npix_y_val,always_apply=True,p=1)])
t_test = A.Compose([A.Resize(npix_x_val,npix_y_val,always_apply=True,p=1)])

#datasets
if (datasettype == "synthetic-moon"):
    train_set = FakeMoonDataset(IMAGE_PATH, MASK_PATH, X_train, t_train, run_name)
    val_set = FakeMoonDataset(IMAGE_PATH, MASK_PATH, X_val, t_val, run_name)
    test_set = FakeMoonDataset(IMAGE_PATH, MASK_PATH, X_test, t_test, run_name)
    removed_set = FakeMoonDataset(IMAGE_PATH, MASK_PATH, X_removed, t_test, run_name)

if (datasettype == "real-moon"):
    train_set = RealMoonDataset(IMAGE_PATH, MASK_PATH, X_train, t_train, run_name)
    val_set = RealMoonDataset(IMAGE_PATH, MASK_PATH, X_val, t_val, run_name)
    test_set = RealMoonDataset(IMAGE_PATH, MASK_PATH, X_test, t_test, run_name)
    removed_set = RealMoonDataset(IMAGE_PATH, MASK_PATH, X_removed, t_test, run_name)

if (datasettype == "ai4mars" or datasettype == "ai4mars-inference"):
    train_set = AI4MarsDataset(IMAGE_PATH, MASK_PATH_TRAIN, RANGE_30M, X_train, t_train, run_name, test = False)
    val_set = AI4MarsDataset(IMAGE_PATH, MASK_PATH_TRAIN, RANGE_30M, X_val, t_val, run_name, test = False)
    test_set = AI4MarsDataset(IMAGE_PATH, MASK_PATH_TEST, RANGE_30M, X_test, t_test, run_name, test = True)
    removed_set = []

if (datasettype == "marsdataset3"):
    train_set = MarsDatasetv3(IMAGE_PATH + 'train/', MASK_PATH + 'train/', SKY_PATH + 'train/', X_train, t_train, run_name)
    val_set = MarsDatasetv3(IMAGE_PATH + 'val/', MASK_PATH + 'val/', SKY_PATH + 'val/', X_val, t_val, run_name)
    test_set = MarsDatasetv3(IMAGE_PATH + 'test/', MASK_PATH + 'test/', SKY_PATH + 'test/', X_test, t_test, run_name)
    removed_set = []


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------

# metrics
metric_names = ["Accuracy","Balanced Accuracy","Jaccard Score"]
datanames = [X_val,X_test,X_train,X_removed]
datatests = [val_set,test_set,train_set,removed_set]
datadirs = ["Validation","Test","Training","Removed"]



for idata,dataloader in enumerate(datatests):

    if idata>1: continue

    figsrun = figsdir + run_name + "/"
    if not os.path.exists(figsrun): os.mkdir(figsrun)

    if os.path.exists(figsrun + datadirs[idata] + "/" + "metrics.txt"):
        f = open(figsrun + datadirs[idata] + "/" + "metrics.txt", "r")
        print("  ")
        print(datadirs[idata]+":")
        print(f.read())

    else:


        dataset = datatests[idata]
        if (not dataset): continue

        acc_score=[]
        balacc_score=[]
        jacc_score=[]
        conf_matrix=[]

        for i, data in enumerate(tqdm(dataset)):

            #get iteration
            img_orig, img, output_orig, output, mask_orig, mask = data[0], data[1], data[2], data[3], data[4], data[5]

            # Performance metrics
            #need 1d vectors for sklearn.metrics -> need cpu to convert to numpy
            output = output.flatten()
            mask = mask.flatten()

            acc_score.append(metrics.accuracy_score(mask,output))
            balacc_score.append(balanced_accuracy_score(mask,output))
            jacc_score.append(metrics.jaccard_score(mask,output,average = "macro",zero_division = 1))
            conf_matrix.append(metrics.confusion_matrix(mask,output,labels = [0,1,2]))

            # Plotting segmentation results:
            if (not os.path.exists(figsrun + datadirs[idata] + "/")): os.makedirs(figsrun + datadirs[idata] + "/")
            plot_prediction(img_orig,mask_orig,output_orig,datanames[idata][i],balacc_score[i],i,npix_x_val,npix_y_val,figsrun + datadirs[idata] + "/")

        # plot_conf_matrix(conf_matrix,run_name+"_")
        print("Accuracy: {:.10f}".format(np.mean(acc_score)))
        print("Balanced Accuracy: {:.10f}".format(np.mean(balacc_score)))
        print("Jaccard Score: {:.10f}".format(np.mean(jacc_score)))

        mean_scores = [[np.mean(acc_score),np.mean(balacc_score),np.mean(jacc_score)]]
        outfile = figsrun + datadirs[idata] + "/" + "metrics.txt" 
        column_names = [c.replace(' ','_') for c in metric_names]
        df_results = pd.DataFrame(data = mean_scores, columns = column_names)
        df_results.to_csv(outfile, index = False, sep = " ")

















