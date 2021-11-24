import numpy as np

def get_X_y(history_size, dataset):
    '''function that splits train / test sets in X and y'''

    X = []
    y = []

    for i in range(history_size, dataset.shape[0]):
        X.append(dataset[i-history_size:i,:])
        y.append(dataset[i,0])

    return np.array(X), np.array(y)


def inverse_transformer(y, scaler):
  '''function that takes a one-dimensional input array (y_test or y_hat) and inverse transforms it.'''
  y = np.c_[y, np.ones(len(y))]

  y = scaler.inverse_transform(y)

  y= y[:,0]

  return y
