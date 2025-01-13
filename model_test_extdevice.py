import model_library
from model_library import *

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
# GLOBAL VARIABLES


# path to data
image_dir = "data_resized_resfact2/"
modelsdir = "models_externaldevice_vgg19lin/"
resultsdir = "results_lin/"

# data_name = ["SMo","RMo","AI4Mars","MarsDatasetv3"]  #"SMo","RMo","AI4Mars","MarsDatasetv3"
models_name = os.listdir(modelsdir)
param_name = ["Execution Time [s]","Memory [MBytes]","#Pixels","FLOPs [G]"]



#check if cuda is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    device_id = torch.cuda.current_device()
    print("Training on Device: ", device)
    print("Device_id: ", device_id)
    print("GPU: ",torch.cuda.get_device_name(device_id))
# device=torch.device("cpu")


# B/W transformation
black_white = True #transforms it into grayscale
if (black_white): color_channels = 1
else: color_channels = 3


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------

def test_network():

    #no adapters...
    set_seed(42,device)

    evaluate_time = []
    used_ram = []
    used_npixels = []
    flops_count = []
    name=[]

    unet_load = []
    history_load = []
    img=[]
    mean=[]
    std_dev=[]
    data_name=[]
    t=[]
    npixels=[]
    mem_start=[]
    mem_end=[]
    model_name=[]
    model=[]
    img_name=[]


    # mem=used_memory()
    # print(mem)
    # gc.collect()
    # torch.cuda.empty_cache() 
    # mem=used_memory()
    # print(mem)

    list_temp = sorted(os.listdir(image_dir))
    print(list_temp)


    
    
    # Test
    for img_name in list_temp:


        # mem=used_memory()
        # print(mem)
        gc.collect()
        torch.cuda.empty_cache() 
        mem=used_memory()
        # print(mem)

        set_seed(42,device)

        if (img_name==''):
            # print("empty image") 
            continue


        #lettura immagini + normalization
        img = cv2.imread(image_dir + img_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #from BGR to RGB 


        if (img_name.startswith("AI4")): 
            # continue
            mean = [0.2555624275034261, 0.2555624275034261, 0.2555624275034261]
            std_dev = [0.10463496527047811, 0.10463496527047811, 0.10463496527047811]
            for model in models_name:
                if (model.endswith("_Unet_AI4.pt")):
                    model_name = model
                    data_name="AI4Mars"     
                if (model.endswith("_Unetlin_AI4.pt")):
                    model_name = model
                    data_name="AI4Mars_lin"  


        if (img_name.startswith("Mars")): 
            continue
            mean = [0.6096051074840404, 0.5055833881828149, 0.35589452779742]
            std_dev = [0.14303784363117197, 0.12331537394970005, 0.0983118132001638]
            for model in models_name:
                if (model.endswith("RandomCrop_adapter_Unet_Mars.pt")):
                    model_name = model
                    data_name="MarsDatasetv3"
                if (model.endswith("RandomCrop_bnorm_adapter_Unet_Mars.pt")):
                    model_name = model
                    data_name="MarsDatasetv3_bnorm"
                if (model.endswith("adapter_Unetlin_Mars.pt")):
                    model_name = model
                    data_name="MarsDatasetv3_lin"       


        if (img_name.startswith("RMo")):
            continue 
            mean = [0.3748141096978818, 0.35596510164602224, 0.2955574219124973]
            std_dev = [0.218745767389803, 0.2105994343046987, 0.18040353055725178]
            for model in models_name:
                if (model.endswith("RandomCrop_adapter_Unet_RMo.pt")):
                    model_name = model
                    data_name="RMo"
                    # print(model_name)
                if (model.endswith("RandomCrop_bnorm_adapter_Unet_RMo.pt")):
                    model_name = model
                    data_name="RMo_bnorm"     
                if (model.endswith("_Unetlin_RMo.pt")):
                    model_name = model
                    data_name="RMo_lin" 

        if (img_name.startswith("SMo")):
            continue 
            mean = [0.38694046508089297, 0.3868203630007706, 0.38693639003325914]
            std_dev = [0.2525432925470617, 0.25246713776664603, 0.2525414825486369]
            for model in models_name:
                if (model.endswith("SMo.pt")): model_name = model
            data_name="SMo" 



        t = T.Compose([T.ToTensor(), T.Normalize(mean, std_dev), T.Grayscale()])
        img = t(img)
        img = img.to(device)

        print(img_name)
        print(model_name)
        print(modelsdir + model_name)
        print("Evaluating...")




        # loading model
        unet_load, history_load = torch.load(modelsdir + model_name, map_location=device)

        print("pippo")


        # Evaluation
        start = time.time()
        img = img[np.newaxis,:,:,:]
        npixels = (img.size(2),img.size(3))
        
        gc.collect()
        torch.cuda.empty_cache() 
        mem_start = used_memory()
        # print(mem_start)
        _ = unet_load(img)
        mem_end = used_memory()
        # print(mem_end)
        end = time.time()



        # FLOPS counting    
        flops = FlopCountAnalysis(unet_load, img)
        # flops=0

        evaluate_time.append(round(end-start,3))
        used_ram.append(round(mem_end - mem_start,2)) #MByte
        # print(used_ram)
        used_npixels.append(npixels)
        flops_count.append(round(flops.total()/1e+9,2))
        # flops_count.append(flops)
        name.append(data_name)


        # mem=used_memory()
        # print(mem)

        del end, start
        del img
        del mem_end
        del npixels
        del flops
        del unet_load
        del history_load
        del mean
        del std_dev
        # del data_name
        del t
        del mem_start
        del model_name
        del model
        del img_name
        gc.collect()
        torch.cuda.empty_cache() 

        


        
        
       

    results_dict = {'Image name': name, 'Execution Time [s]': evaluate_time, 'Memory [MBytes]': used_ram, '#Pixels': used_npixels, 'FLOPs [G]': flops_count}
    outfile = resultsdir + "/results_vgg19_resfact1_JNano_adapt_"+data_name+".txt"
    df_results = pd.DataFrame(results_dict)
    df_results.to_csv(outfile, sep = "\t",index=False)

    gc.collect()

#--------------------------------------------
#--------------------MAIN--------------------
#---------------------------------------------

test_network()

