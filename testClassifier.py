from keras.preprocessing.image import img_to_array
import numpy as np
from keras.models import model_from_json
import cv2

modelJSON= open('classifier.json','r')
loadedModelJSON=modelJSON.read()
modelJSON.close()
model=model_from_json(loadedModelJSON)
model.load_weights('classifierWeights.h5')
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

testImage=cv2.imread('trail4.jpg')
orig=testImage.copy()

testImage=cv2.resize(testImage, (64,64))
testImage= testImage.astype('float')/255.0
testImage= img_to_array(testImage)
testImage= np.expand_dims(testImage,axis=0)
(notHelmetImage, helmetImage) = model.predict(testImage)[0]

label= 'Helmet' if helmetImage > notHelmetImage else 'notHelmet'
proba = helmetImage if helmetImage > notHelmetImage else notHelmetImage
label = '{}: {:.2f}%'.format(label,proba*100)

cv2.putText(orig, label, (10,25), cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 255, 0), 2)

cv2.imshow("Output", orig)
cv2.waitKey(0)
