import matplotlib
matplotlib.use('Agg')
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from imutils import paths
from sklearn.model_selection import train_test_split
from pyimagesearch.lenet import LeNet
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import os

EPOCHS = 25
INIT_LR = 1e-3
BS = 32

data=[]
labels=[]

imagePaths=sorted(list(paths.list_images('images')))
random.seed(42)
random.shuffle(imagePaths)

for imagePath in imagePaths:
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (64,64))
    image = img_to_array(image)
    data.append(image)
    label= imagePath.split(os.path.sep)[-2]
    label=1 if label=='Helmet' else 0
    labels.append(label)

data=np.array(data, dtype='float')/255.0
labels=np.array(labels)

trainHelmet, validationHelmet, trainLabel, validationLabel = train_test_split(data, labels, test_size=0.25, random_state=42)

trainLabel=to_categorical(trainLabel, num_classes=2)
validationLabel=to_categorical(validationLabel, num_classes=2)

aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')

model = LeNet.build(width=64, height=64, depth=3, classes=2)
opt = Adam(lr=INIT_LR, decay=INIT_LR/EPOCHS)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

H= model.fit_generator(aug.flow(trainHelmet,trainLabel, batch_size=BS),validation_data=(validationHelmet,validationLabel),steps_per_epoch=len(trainHelmet)//BS, epochs=EPOCHS, verbose=1)

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.title("Training Loss and Accuracy on Santa/Not Santa")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="lower left")
plt.savefig('graph.jpg')


classifierJSON=model.to_json()
with open("classifier.json", "w") as json_file:
    json_file.write(classifierJSON)
model.save_weights('classifierWeights.h5')





