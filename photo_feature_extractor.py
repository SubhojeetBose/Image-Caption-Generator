from os import listdir
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from pickle import dump

def image_features(directory):
    #pretrained model
    model = VGG16()
    #removing the last classification layer
    model.layers.pop()
    model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
    #model info
    print(model.summary())
    features = dict()
    #extracting features from each image
    cnt = 0
    for name in listdir(directory):
        #extracting and changing img to array
        file = directory + '/' + name
        img = load_img(file, target_size=(224,224))
        img = img_to_array(img)
        #reshape the array according to the model
        img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
        #preprocess image for VGG Model
        img = preprocess_input(img)
        #get and store features
        img_id = name.split('.')[0]
        feature = model.predict(img,verbose=0)
        features[img_id] = feature
        cnt = cnt + 1
        print('>%d   %s' %(cnt, name))
    return features

#calling the function to extract all the features
directory = 'Flicker8k_Dataset'
features = image_features(directory)
print('Image Features: %d' %len(features))
#save to file
dump(features, open('features.pkl', 'wb'))
