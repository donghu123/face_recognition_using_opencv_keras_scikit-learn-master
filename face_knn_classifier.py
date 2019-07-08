

from keras.models import load_model
from load_face_dataset import load_dataset, IMAGE_SIZE, resize_image


import numpy as np


from fr_utils import img_to_encoding
from sklearn.model_selection import cross_val_score, ShuffleSplit, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib

import matplotlib.pyplot as plt

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


from keras.utils import CustomObjectScope
import tensorflow as tf
with CustomObjectScope({'tf': tf}):
    facenet = load_model('./model/nn4.small2.v1.h5')

class Dataset:
    
    def __init__(self, path_name): 
        
        self.X_train = None
        self.y_train = None
        
       
        self.path_name = path_name
    
   
    def load(self, img_rows = IMAGE_SIZE, img_cols = IMAGE_SIZE, img_channels = 3, model = facenet):
        
        images, labels = load_dataset(self.path_name)
        
        X_embedding = img_to_encoding(images, model)
       
        print('X_train shape', X_embedding.shape)
        print('y_train shape', labels.shape)
        print(X_embedding.shape[0], 'train samples')
       
        self.X_train = X_embedding
        self.y_train = labels


class Knn_Model:
    
    def __init__(self):
        self.model = None
    def cross_val_and_build_model(self, dataset):
        
        k_range = range(1,31)
       
        k_scores = []
        print("k vs accuracy:")
        for k in k_range:
            knn = KNeighborsClassifier(n_neighbors = k)
            cv = KFold(n_splits = 10, shuffle = True, random_state = 0)
           
            score = cross_val_score(knn, dataset.X_train, dataset.y_train, cv = 10, scoring = 'accuracy').mean()
            k_scores.append(score)
            print(k, ":", score)
     
        plt.plot(k_range, k_scores)
        
        plt.title("KNN") #fontsize = 24)
        plt.xlabel('Value of K for KNN')#, fontsize = 14)
        plt.ylabel('Cross-Validated Accuracy')#, fontsize = 14)
        plt.tick_params(axis='both')#, labelsize = 14)
        plt.show()
        n_neighbors_max = np.argmax(k_scores) + 1
        print("The best k is: ", n_neighbors_max)
        print("The accuracy is: ", k_scores[n_neighbors_max - 1], "When n_neighbor is: ", n_neighbors_max)
        
        self.model = KNeighborsClassifier(n_neighbors = n_neighbors_max)
        
       
    def train(self, dataset):
        self.model.fit(dataset.X_train, dataset.y_train)
    def save_model(self, file_path):
        #save model
        joblib.dump(self.model, file_path)
    def load_model(self, file_path):
        self.model = joblib.load(file_path)
    def predict(self, image):
        image = resize_image(image)
        image_embedding = img_to_encoding(np.array([image]), facenet)
        label = self.model.predict(image_embedding)
        return label[0]


if __name__ == "__main__":
    dataset = Dataset('dataset')
    dataset.load()
    
    model = Knn_Model()
    model.cross_val_and_build_model(dataset)

    model.train(dataset)
    model.save_model('model/knn_classifier.model')
