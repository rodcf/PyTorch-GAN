from keras.datasets import mnist
from sklearn.model_selection import train_test_split
import numpy as np

from src.crosslid.FeatureCNN import get_deep_features
from src.crosslid.Model import get_fake_images
from src.crosslid.utilities import compute_crosslid


def get_real_images(size):
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train.astype('float32')
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
    sel_pct = size/X_train.shape[0]
    if size < X_train.shape[0]:
        (X_train, xtr2, ytr1, ytr2) = train_test_split(X_train, y_train, train_size=sel_pct, stratify=y_train)
    return X_train

class CrossLID:
    
    def __init__(self, sample_size=200000):
        self.sample_size = sample_size
        X_real = get_real_images(sample_size)
        self.X_real_features = get_deep_features(X_real)
    

    def calculate_cross_lid(self, onnx_path: str = ""):
        X_fake = get_fake_images(onnx_path, self.sample_size)
        X_fake = X_fake.reshape((self.sample_size,28,28,1))
        X_fake_features = get_deep_features(X_fake)
        cross_lid = compute_crosslid(X_fake_features, self.X_real_features, k=100, batch_size=1000)
        print('CrossLID score: ', cross_lid)
        return cross_lid



