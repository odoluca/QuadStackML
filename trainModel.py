import os.path

import numpy as np
import tensorflow as tf
from keras.src.backend import shape, convert_to_tensor
from tensorflow import convert_to_tensor,device
import prepModel2 as prepModel
import prepData
import random
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import pickle as pkl


if __name__=="__main__":
    #TRAIN PARAMETERS
    num_epoch = 10 #BEST 25
    splitRatio = 0.7 #ratio of train:test
    batch_size = 128
    initial_lr = 1e-3 #initial learning rate. used in modelbuild
    lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='loss',
        factor=0.5,
        patience=2,
        verbose=1,
        min_lr=1e-6,
        min_delta= 1e-5
    )

    #MODEL PARAMETERS
    fc_units = (128, 64, 32)
    filter_sizes = (80,80,96,96)
    kernel_sizes = (4,12,24,36)
    opt_func = "Adam"

    model=prepModel.build_model((5,100),initial_lr, fc_units,filter_sizes,kernel_sizes,opt_func)
    print("model is created.")
    model.summary()
    tf.keras.utils.plot_model(model, show_shapes=True)

    if (not os.path.isfile("data.pkl")):


        dirList=[("/home/osman/PycharmProjects/TFproject/BG4pos",1.0,False),("/home/osman/PycharmProjects/TFproject/BG4neg",0.0,False)]
        # dirList=[("/home/osman/PycharmProjects/TFproject/BG4neg",0.0)]


        allX=np.empty(shape=(0,5,100) , dtype=np.float32)
        allY=np.empty(shape=(0,)  , dtype=np.float32)
        for dirName,defY,useReverse in dirList:
            print("Preparing", dirName, "...")
            dataX,dataY=prepData.PrepDir(dirName, defaultY=defY, maxFilesToProcess=-1, includeOtherStrand=useReverse)
            # print(data)
            print(dirName, "complete. Size:", dataY.shape[0])
            allX=np.concatenate((allX,dataX))
            allY=np.concatenate((allY,dataY))

        # Zip the lists together and shuffle
        print("Shuffling data...")
        combined = list(zip(allX, allY))
        random.shuffle(combined)

        #split lists
        print("Files analyzed. Splitting data. Split by ",splitRatio*100,"/",(1-splitRatio)*100)
        cutIndex=int(len(combined)*splitRatio)
        train_combined=combined[:cutIndex]
        test_combined=combined[cutIndex:]
        trainX,trainY = zip(*train_combined)
        testX,testY=zip(*test_combined)

        pkl.dump((trainX,trainY,testX,testY),open("data.pkl","wb"))
    else:
        print("Loading data file data.pkl...")
        trainX, trainY, testX, testY= pkl.load(open("data.pkl","rb"))

    print(f"training size {len(trainY)}")
    print(f"test size {len(testY)}")

    print("Converting back to tensor")
    with device('/cpu:0'):
        # x = tf.convert_to_tensor(x, np.float32)
        # y = tf.convert_to_tensor(y, np.float32)

        trainX=convert_to_tensor(np.array(trainX), np.float32)
        trainY=convert_to_tensor(np.array(trainY), np.float32)
        testX=convert_to_tensor( np.array(testX) , np.float32)
        testY=convert_to_tensor( np.array(testY) , np.float32)

    print("Training started")
    print(trainX.shape,trainY.shape)
    model.fit(trainX,trainY,epochs=num_epoch, batch_size=batch_size, callbacks=[lr_schedule] )
    print("Training complete. starting evaluation using test set")
    evaluation=model.evaluate(x=testX,y=testY)
    print("Accuracy and loss:", evaluation)

    def compare(index):
        prediction=model.predict([allX[index:index+1]])
        print("prediction:",prediction,"true:",allY[index:index+1])

    def calculate_auc(testX, testY, model,verbose=False):
        predY = model.predict(testX).ravel()
        auc_metric = tf.keras.metrics.AUC()
        auc_metric.update_state(testY, predY)
        auc = auc_metric.result().numpy()
        fpr, tpr, thresholds = roc_curve(testY, predY)
        if verbose:
            print("False Positive Rates:", fpr)
            print("True Positive Rates:", tpr)
            print("Thresholds:", thresholds)
        return auc,fpr,tpr,thresholds

    # print(calculate_auc(testX,testY,model))
    def plot_roc_curve(testX,testY,model,filename='roc_curve.png'):
        # predY = model.predict(testX).ravel()
        # fpr, tpr, thresholds = roc_curve(testY, predY)
        # roc_auc = auc(fpr, tpr)
        auc,fpr,tpr,thresholds=calculate_auc(testX,testY,model)
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc='lower right')
        plt.grid(True)
        # plt.show()
        plt.savefig(filename)
        return auc,fpr,tpr,thresholds

    auc,fpr,tpr,thr=plot_roc_curve(testX,testY,model)
    print("Area under curve:",auc)
    print("Plot is saved as png")


    def lr_finder(model, x, y, start_lr=1e-7, end_lr=1e-2, batch_size=128, num_batches=100):
        x= x.numpy()
        y = y.numpy()
        lrs = np.geomspace(start_lr, end_lr, num_batches)
        losses = []
        weights = model.get_weights()  # Save initial weights
        for i, lr in enumerate(lrs):
            # ✅ Handle different TF versions
            if hasattr(model.optimizer.learning_rate, 'assign'):
                model.optimizer.learning_rate.assign(lr)
            else:
                model.optimizer.learning_rate = lr
            # Sample a batch
            idx = np.random.randint(0, x.shape[0], batch_size)
            loss = model.train_on_batch(x[idx], y[idx])[0]
            losses.append(loss)
            print(f"Step {i + 1}/{num_batches} - lr: {lr:.6f} - loss: {loss:.4f}")
        model.set_weights(weights)  # Restore original weights
        # Plotting
        plt.figure(figsize=(8, 5))
        plt.plot(lrs, losses)
        plt.xscale('log')
        plt.xlabel("Learning Rate")
        plt.ylabel("Loss")
        plt.title("Learning Rate Finder")
        plt.grid(True)
        plt.savefig("lr_finder_plot.png")
        plt.show()


    lr_finder(model, trainX, trainY, batch_size=batch_size)
