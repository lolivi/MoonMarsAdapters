import os
import numpy as np
import cv2

IMAGE_PATH = "ai4mars-dataset-merged-0.1/msl/images/edr/"
MASK_PATH_TRAIN = "ai4mars-dataset-merged-0.1/msl/labels/train/"
MASK_PATH_TEST = "ai4mars-dataset-merged-0.1/msl/labels/test/masked-gold-min3-100agree/"
MASK_ROVER = "ai4mars-dataset-merged-0.1/msl/images/mxy/"
RANGE_30M = "ai4mars-dataset-merged-0.1/msl/images/rng-30m/"

def remove_rover(path):
    
    for dirname, _, filenames in os.walk(path):
        for filename in filenames:
            
            imname = filename.split('.')[0]
            imname_mask = imname #_merged + EDR indica mask .png

            imname = imname.replace("_merged","")
            imname_image = imname #senza _merged indica immagine .JPG

            imname_rover = imname.replace("EDR","MXY") #senza _merged e EDR sostituito da MXY indica rover .PNG
            imname_rng = imname.replace("EDR","RNG") #senza _merged e EDR sostituito da RNG indica rover .PNG

            #eliminating all images with rover
            rover_raw = np.array(cv2.imread(MASK_ROVER + imname_rover + ".png"))

            rover = np.zeros((1024,1024))
            rover[:,:] = rover_raw[:,:,0]
            if ((rover == 1).any()): 
                print("Removing ", imname_image)
                os.remove(IMAGE_PATH + imname_image + ".JPG")
                os.remove(path + imname_mask + ".png")
                os.remove(MASK_ROVER + imname_rover + ".png")
                os.remove(RANGE_30M + imname_rng + ".png")

remove_rover(MASK_PATH_TRAIN)
remove_rover(MASK_PATH_TEST)