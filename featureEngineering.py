from Preprocessing import Preprocess
import numpy as np

import tensorflow as tf
keras = tf.keras
from keras.utils import to_categorical

from sklearn.preprocessing import StandardScaler
import sklearn.model_selection as ms

class FeatureEngineer:
    def __init__(self, feature, path, train, test):
        self.preprocessor = Preprocess(feature, path, train, test)
        self.train_feat, self.test_feat = self.preprocessor.combine_data(self.preprocessor.features)
        self.df_train = self.preprocessor.df_train
        self.df_test = self.preprocessor.df_test


    def convert_labels(self):
        #dictionary to change labels to integers
        levels = {key: value \
            for key, value in zip(self.preprocessor.df_train.word.unique(), \
            range( len( self.df_train.word.unique() )) ) }

        #binary encoding labels
        labels = to_categorical(np.array([levels[key] \
            for key in self.df_train["word"]], dtype=np.float32))

        return labels, levels

    def scale(self):
        size_train = len(self.train_feat)
        size_test = len(self.test_feat)
        sc = StandardScaler()

        for i in range(size_train):
            self.train_feat[i] = sc.fit_transform(self.train_feat[i])
        
        for i in range(size_test):
            self.test_feat[i] = sc.fit_transform(self.test_feat[i])

    def remake_array(self, arr):
      ''' To ensure the last axis is the same for all samples '''
      remade_array = np.zeros((len(arr), 99,13))
      for i, item in enumerate(arr):
        for j in range(item.shape[0]):
          for k in range(len(item[j])):
            remade_array[i][j][k] += arr[i][j][k]

      return remade_array[:, :, :, np.newaxis]


    def define_splitting(self, feat, lbl, ratio):
      np.random.seed(37555)
      X_1 , X_2, y_1 , y_2 = ms.train_test_split(feat, lbl, test_size=ratio)

      return X_1 , X_2, y_1 , y_2

    def splitting(self):
      remade_data = self.remake_array(self.train_feat)
      labels, _ = self.convert_labels()

      X_train , X_2, y_train , y_2 = self.define_splitting(remade_data, labels, 0.3)
      X_val , X_test, y_val , y_test = self.define_splitting(X_2, y_2, 0.5)

      return (X_train, y_train), (X_val, y_val), (X_test, y_test)
