import synthetic_moon_lib
from synthetic_moon_lib import *

#dataframe
df = create_df()
print('Total Images: ', len(df))

#plotting example
plotinput(1,df)

#train-test-splitting
X_trainval, X_test = train_test_split(df['id'].values, test_size = 0.1, random_state=42, shuffle = True)
X_train, X_val = train_test_split(X_trainval, test_size = 0.1116, random_state=42, shuffle = True)
print('Train Size   : ', len(X_train))
print('Val Size     : ', len(X_val))
print('Test Size    : ', len(X_test))

#mean and standard deviations
#mean, std = getmeanstd(X_train)

#data augmentation and dataloaders
mean = [0.3840956136340923, 0.38397682789179083, 0.3840915669371818]
std = [0.25272097997268805, 0.2526449765088852, 0.25271917037309793]

#npix_x, npix_y = 480,704
npix_x, npix_y = 224,352

t_train = A.Compose([A.HorizontalFlip(p=0.5), 
                     A.Blur(p=0.2), #possible dust (very rare)
                     A.RandomScale(scale_limit = (0,0.1), p = 0.5), #possible distortions due to zooming (causing errors in shape...)
                     A.RandomBrightnessContrast(0.5,0.5,p = 0.5), #frequent changes because of illumination 
                     A.GaussNoise(10,p=0.3), #interference (rare)
                     A.RandomCrop(224,352,p=0.3), #crop of half the image with same ratio,
                     #A.Resize(480,704,always_apply=True,p=1) #unet needs images divisible by 32
                     A.Resize(npix_x,npix_y,always_apply=True,p=1)
                    ])

#no augmentation and mean and std devs are from training as we shouldn't know the distribution of test set in real case scenario
t_val = A.Compose([A.Resize(npix_x,npix_y,always_apply=True,p=1)])
t_test = A.Compose([A.Resize(npix_x,npix_y,always_apply=True,p=1)])

#datasets
train_set = FakeMoonDataset(IMAGE_PATH, MASK_PATH, X_train, mean, std, t_train)
val_set = FakeMoonDataset(IMAGE_PATH, MASK_PATH, X_val, mean, std, t_val)
test_set = FakeMoonDataset(IMAGE_PATH, MASK_PATH, X_test, mean, std, t_test)

#dataloader
batch_size = 16
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)   
test_loader = DataLoader(test_set)  

#frequency distribution
#train_count, val_count, test_count = getdistcts(train_set,val_set,test_set)

train_count = [0.0712635,  0.17793552, 0.75080098]
val_count = [0.06929933, 0.20140098, 0.72929969]
test_count = [0.06974799, 0.19097118, 0.73928084]

rocks_count = [train_count[0],val_count[0],test_count[0]]
sky_count = [train_count[1],val_count[1],test_count[1]]
terrain_count = [train_count[2],val_count[2],test_count[2]]
class_weights = torch.tensor([1./train_count[0],1./train_count[1],1./train_count[2]],device = device, dtype=torch.float)

index = ['Train','Validation','Test']
df_frequency = pd.DataFrame({'Rocks': rocks_count, 'Sky': sky_count, 'Terrain': terrain_count},index)

df_frequency.plot.bar(stacked = True)
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
plt.show()

#metrics
metric_names = ["Accuracy","Balanced Accuracy","Jaccard Score"]

#no adapters...
unet = smp.Unet('resnet18', decoder_channels = (256, 128, 64, 32, 16), classes = 3, activation = None, in_channels = color_channels)
unet_train = unet

#max_lr = 1e-2
max_lr = 1e-3 #best one!
#max_lr = 1e-4 

epoch = 3
weight_decay = 1e-4

criterion = nn.CrossEntropyLoss(class_weights)
optimizer_encoder = torch.optim.AdamW(unet_train.get_submodule("encoder").parameters(), lr=max_lr, weight_decay=weight_decay)
optimizer_decoder = torch.optim.AdamW(unet_train.get_submodule("decoder").parameters(), lr=max_lr, weight_decay=weight_decay)
optimizer_seghead = torch.optim.AdamW(unet_train.get_submodule("segmentation_head").parameters(), lr=max_lr, weight_decay=weight_decay)
#optimizer_encoder = torch.optim.SGD(unet_train.get_submodule("encoder").parameters(), lr=max_lr/5., momentum = 0.9)
#optimizer_decoder = torch.optim.SGD(unet_train.get_submodule("decoder").parameters(), lr=max_lr, momentum = 0.9)
#optimizer_seghead = torch.optim.SGD(unet_train.get_submodule("segmentation_head").parameters(), lr=max_lr, momentum = 0.9)

optimizer = [optimizer_encoder,optimizer_decoder,optimizer_seghead]

#optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=weight_decay)
'''
sched_encoder = torch.optim.lr_scheduler.ExponentialLR(optimizer_encoder, gamma = 0.93) #0.1 factor every 30 epochs
sched_decoder = torch.optim.lr_scheduler.ExponentialLR(optimizer_decoder, gamma = 0.93)
sched_seghead = torch.optim.lr_scheduler.ExponentialLR(optimizer_seghead, gamma = 0.93)
sched = [sched_encoder,sched_decoder,sched_seghead]
'''

sched_encoder = None
sched_decoder = None
sched_seghead = None
sched = [sched_encoder,sched_decoder,sched_seghead]

'''
sched_encoder = torch.optim.lr_scheduler.StepLR(optimizer_encoder, step_size = 3, gamma = 0.79)
sched_decoder = torch.optim.lr_scheduler.StepLR(optimizer_decoder, step_size = 3, gamma = 0.79)
sched_seghead = torch.optim.lr_scheduler.StepLR(optimizer_seghead, step_size = 3, gamma = 0.79)
sched = [sched_encoder,sched_decoder,sched_seghead]
'''

history = fit(epoch, unet_train, train_loader, val_loader, criterion, optimizer, sched, metric_names)

#saving model
torch.save((unet_train,history),'models/Unet_best.pt')

#plotting results
plot_loss(history)
plot_score(history,metric_names)
plot_lrs(history)
