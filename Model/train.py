import argparse

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf 
from tensorflow.keras.models import load_model 
from sklearn.metrics import classification_report ,confusion_matrix
from sklearn.utils.class_weight import compute_class_weight


from model import create_model
from preprocess_data import process_labels , augment_data



def show_loss_plot(history):
    
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy/Loss")
    plt.title("Model Training Loss")
    
    plt.legend()
    plt.show()


def show_accuracy_plot(history):
    
    plt.plot(history.history["accuracy"], label="accuracy")
    plt.plot(history.history["val_accuracy"], label="val_accuracy")
    plt.title("Model Training Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim([0, 1])
    
    plt.legend(loc="lower right")
    plt.show()


def evaluate_model(test_data):
    
    model = load_model("./emotion_model.h5")
    xTest, yTest = test_data
    y_pred = model.predict(xTest)
    
    y_test_labels = np.argmax(yTest, axis=1)
    y_pred_labels = np.argmax(y_pred, axis=1)

    print(classification_report(y_test_labels, y_pred_labels))
    
    print(confusion_matrix(y_test_labels, y_pred_labels))


def train(args):
    train_data , test_data , X , y = process_labels(args.root)
    xTrain , yTrain = train_data
    xTest , yTest = test_data 
    xTrain , yTrain = augment_data(xTrain , yTrain)
    model = create_model(input_shape=(X.shape[1], 1), num_classes=y.shape[1])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5),
    ]
    
    class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(yTrain.argmax(axis=1)),
    y=yTrain.argmax(axis=1)
    )
    
    class_weights_dict = dict(enumerate(class_weights))
    model.summary()
    # Train the model
    history = model.fit(
        xTrain,
        yTrain,
        validation_data=(xTest, yTest),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        class_weight=class_weights_dict,
    )
    
    model.save(args.checkpoint_dir)
    if args.debug:
        evaluate_model((xTest , yTest))
        show_accuracy_plot(history)
        show_loss_plot(history)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser('Train the Emotion Recognition System')
    
    parser.add_argument(
        '--epochs',
        required= True,
        type= int,
        help= 'Number of epochs'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help= 'Showing the loss and accuracy plot'
    )
    
    parser.add_argument(
        '--root', 
        required= True,
        help= 'The path for the csv file (dataset)'
    )
    
    parser.add_argument(
        '--checkpoint_dir',
        default= './emotion_model.h5',
        help= 'path to save the model in it'
    )
    
    parser.add_argument(
        '--batch_size' ,
        default= 8 ,
        type = int , 
        help='The batch size'
        )
    
    args = parser.parse_args()
    train(args)
