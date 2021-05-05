import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

'''
PetImages dataset link: https://www.microsoft.com/en-us/download/details.aspx?id=54765
Thanks to Microsoft
'''

REBUILD_DATA = True

class DogsVSCats():
    # resize the img
    IMG_SIZE = 50
    CATS = "dataset/PetImages/Cat"
    DOGS = "dataset/PetImages/Dog"
    LABELS = {CATS: 0, DOGS: 1}

    training_data = []
    catcount = 0
    dogcount = 0

    #preprocessing data
    def make_training_data(self):
        for label in self.LABELS:

            # iterate all of the image in the directory
            # visualize the progress bar using tqdm
            # interate each picture of the current label
            for ind in tqdm(os.listdir(label)):
                try:  # since some of the image are not good to resize
                    path = os.path.join(label, ind)
                    # colors do not matter between dogs and cats so we can convert to grayscale
                    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))

                    # encode scalar label to be 1-hot vector label using np.eye()
                    self.training_data.append([np.array(img), np.eye(2)[self.LABELS[label]]])

                    # should be very close
                    if label == self.CATS:
                        self.catcount += 1
                    elif label == self.DOGS:
                        self.dogcount += 1
                except Exception as e:
                    pass



        np.random.shuffle(self.training_data)
        np.save("training_data.npy", self.training_data)
        print("Cats: ", self.catcount)
        print("Dogs: ", self.dogcount)



if REBUILD_DATA:
    dogsvcats = DogsVSCats()
    dogsvcats.make_training_data()
    #plt.imshow(dogsvcats.training_data[1][0], cmap = "gray") if you want to visualize the image



