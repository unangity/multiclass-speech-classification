import numpy as np
import pandas as pd

class Preprocess:
    def __init__(self, feature, paths, train, test):

        self.features = np.load(feature, allow_pickle = True)
        self.path = np.load(paths)
        self.train = pd.read_csv(train)
        self.test = pd.read_csv(test)

    def balance_arrays(self, feat):
        '''function to make all arrays same dimensions by zero padding the arrays that 
    are less than the largest array. Part of preprocessing'''

        ## get dimension of biggest numpy array in features
        balanced_array = np.copy(feat)
        biggest_shape = max([   i.shape for i in balanced_array  ])

        ## remake arrays
        for i, feature in enumerate(balanced_array):
            if feature.shape[0] < biggest_shape[0]:
                rows_to_add = biggest_shape[0] - feature.shape[0]
                with_zeroes = np.vstack(( feature, np.zeros((rows_to_add, biggest_shape[1])) ))
                balanced_array[i] = with_zeroes

        return balanced_array

    def removing_outliers(self, feat, val = 99):
        clean_features = self.balance_arrays(np.array([i for i in feat if i.shape[0] <= val]))

        return clean_features

    def combine_data(self, feat):
        clean_features = self.removing_outliers(feat)
        dfFeat = pd.DataFrame (clean_features, columns=['features'])
        dfPath = pd.DataFrame(self.path, columns=['path'])

        FeaturePath = dfPath.join(dfFeat, how='right')

        self.df_train = pd.merge(FeaturePath, self.train, on='path') 
        self.df_test = pd.merge(FeaturePath, self.test, on='path')

        train_feat = self.df_train['features'].values 
        test_feat = self.df_test['features'].values

        return train_feat, test_feat