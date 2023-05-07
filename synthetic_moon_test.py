import synthetic_moon_lib
from synthetic_moon_lib import *

#dataframe
df = create_df()
print('Total Images: ', len(df))

#plotting example
plotinput(14,df)

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

plt.figure()
df_frequency.plot.bar(stacked = True)
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
plt.show()

#metrics
metric_names = ["Accuracy","Balanced Accuracy","Jaccard Score"]

# loading model
unet_load,history_load = torch.load('models/Unet_best.pt')

plot_loss(history_load)
plot_score(history_load,metric_names)
plot_lrs(history_load)

scores, conf_matrix = evaluate(unet_load,test_loader,metric_names)

plt.figure()
sns.heatmap(pd.DataFrame(conf_matrix.tolist(),columns = ["Rocks","Sky","Terrain"],index = ["Rocks","Sky","Terrain"]), annot = True)
plt.ylabel("Ground Truth")
plt.xlabel("Predicted Labels")
plt.show()

acc = scores[0]
bal_acc = scores[1]
jacc_score = scores[2]

print("Accuracy: {:.2f}".format(np.mean(acc)))
print("Balanced Accuracy: {:.2f}".format(np.mean(bal_acc)))
print("Jaccard Score: {:.2f}".format(np.mean(jacc_score)))

ibest = bal_acc.index(max(bal_acc))
iworst = bal_acc.index(min(bal_acc))
irandom = random.randint(0,len(test_loader)-1)

image_best = test_set[ibest][0]
image_worst = test_set[iworst][0]
image_rndm = test_set[irandom][0]

mask_best = test_set[ibest][1]
mask_worst = test_set[iworst][1]
mask_rndm = test_set[irandom][1]
        
predicted_best = predicted_mask(unet_load,image_best.view(1,1,npix_x,npix_y))
predicted_worst = predicted_mask(unet_load,image_worst.view(1,1,npix_x,npix_y))
predicted_random = predicted_mask(unet_load,image_rndm.view(1,1,npix_x,npix_y))

#best prediction
fig, ax = plt.subplots(1,2, figsize=(20,10))

ax[0].imshow(image_best.permute(1, 2, 0), cmap = "gray")
ax[0].imshow(mask_best, alpha = 0.5, cmap = "brg", vmin = 0, vmax = 2)
ax[0].set_title('Image with Ground Truth Prediction')

ax[1].imshow(image_best.permute(1, 2, 0), cmap = "gray")
ax[1].imshow(predicted_best.cpu().view(npix_x,npix_y), alpha = 0.5, cmap = "brg",vmin = 0, vmax = 2)
ax[1].set_title('Image with U-Net Prediction (Best)')

#worst prediction
fig, ax = plt.subplots(1,2, figsize=(20,10))

ax[0].imshow(image_worst.permute(1, 2, 0), cmap = "gray")
ax[0].imshow(mask_worst, alpha = 0.5,cmap = "brg", vmin = 0, vmax = 2)
ax[0].set_title('Image with Ground Truth Prediction')

ax[1].imshow(image_worst.permute(1, 2, 0), cmap = "gray")
ax[1].imshow(predicted_worst.cpu().view(npix_x,npix_y), alpha = 0.5,cmap = "brg", vmin = 0, vmax = 2)
ax[1].set_title('Image with U-Net Prediction (Worst)')

#random prediction
fig, ax = plt.subplots(1,2, figsize=(20,10))

ax[0].imshow(image_rndm.permute(1, 2, 0),cmap="gray")
ax[0].imshow(mask_rndm, alpha = 0.5, cmap = "brg", vmin = 0, vmax = 2)
ax[0].set_title('Image with Ground Truth Prediction')

ax[1].imshow(image_rndm.permute(1, 2, 0),cmap="gray")
ax[1].imshow(predicted_random.cpu().view(npix_x,npix_y), alpha = 0.5, cmap = "brg", vmin = 0, vmax = 2)
ax[1].set_title('Image with U-Net Prediction (Random)')

plt.show()