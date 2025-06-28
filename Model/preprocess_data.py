import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder 
import numpy as np
import warnings
# Ignore all warnings
warnings.filterwarnings("ignore")

def process_labels (path_to_csv):
    
    df = pd.read_csv(path_to_csv)

    # Preprocess data
    X = df.drop('class', axis=1).values
    y = df['class'].values
    
    # Ensure features are properly normalized
    X = (X - np.mean(X)) / np.std(X)
    
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    
    onehot_encoder = OneHotEncoder(sparse_output=False)
    y = onehot_encoder.fit_transform(y.reshape(-1, 1))

    # Reshape X for Conv1D
    X = X.reshape(X.shape[0], X.shape[1], 1)

    xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.1, random_state=42)
    
    # Save the label encoder
    np.save('./labels.npy', label_encoder.classes_, allow_pickle=True)

    return (xTrain , yTrain) , (xTest , yTest) , X , y


def augment_data(X, y, n_augmentations=2):
    X_aug = []
    y_aug = []
    
    for _ in range(n_augmentations):
        # Add Gaussian noise
        X_aug.append(X + 0.05 * np.random.randn(*X.shape))
        # Time shifting
        shift = np.random.randint(-5, 5)
        X_aug.append(np.roll(X, shift, axis=1))
        # Pitch scaling
        scale = np.random.uniform(0.9, 1.1)
        X_aug.append(X * scale)
        y_aug.extend([y] * 3)  # Repeat labels

    return np.concatenate(X_aug), np.concatenate(y_aug)
