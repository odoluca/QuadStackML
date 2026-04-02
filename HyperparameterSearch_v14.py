import os.path
import numpy as np
import tensorflow as tf
from keras.src.backend import shape, convert_to_tensor
from keras.src.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow import convert_to_tensor,device
import prepModel14 as prepModel
import prepData
import random
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import pickle as pkl
import keras_tuner as kt

"""CHANGES:
1. learning rate drop is bound to AUC metric
2. also included early stopping
3. concat true or false gives almost no difference. except concat is smaller in size
"""



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

def plot_roc_pr_curves(testX, testY, model, filenamebase='roc_pr_curves'):
    from sklearn.metrics import precision_recall_curve, average_precision_score

    # Reuse your existing AUC helper
    auc_val, fpr, tpr, thresholds = calculate_auc(testX, testY, model)

    # Predictions for PR curve
    predY = model.predict(testX).ravel()
    y_true = np.array(testY).ravel()

    precision, recall, pr_thresholds = precision_recall_curve(y_true, predY)
    ap = average_precision_score(y_true, predY)

    # No-skill baseline for PR is the positive rate
    pos_rate = (y_true > 0.5).mean() if y_true.dtype != bool else y_true.mean()

    # Plot side-by-side
    plt.figure(figsize=(10, 4))

    # ROC
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, label=f'ROC (AUC = {auc_val:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    # Precision-Recall
    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, label=f'PR (AP = {ap:.3f})')
    plt.plot([0, 1], [pos_rate, pos_rate], 'k--', label=f'No-skill (p = {pos_rate:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision–Recall Curve')
    plt.legend(loc='lower left')
    plt.grid(True)
    plt.xlim(0, 1)   # <-- force 0..1 on X (recall)
    plt.ylim(0, 1)   # optional but keeps scale consistent

    plt.tight_layout()
    plt.savefig(f"{filenamebase}_{auc_val:.3f}.png")
    plt.close()

    return auc_val, fpr, tpr, thresholds, precision, recall, pr_thresholds


def lr_finder(model, x, y, start_lr=1e-7, end_lr=1e-1, batch_size=128, num_batches=100):
    x= x.numpy()
    y = y.numpy()
    lrs = np.geomspace(start_lr, end_lr, num_batches)
    losses = []
    weights = model.get_weights()  # Save initial weights
    best_lr=None
    lowest_loss=99999
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
        if (loss<lowest_loss):
            best_lr=lr
            lowest_loss=loss
        print(f"Step {i + 1}/{num_batches} - lr: {lr:.6f} - loss: {loss:.4f}")


    model.set_weights(weights)  # Restore original weights

    #set best LR
    print("Best LR is found: ",best_lr)
    # ✅ Handle different TF versions
    if hasattr(model.optimizer.learning_rate, 'assign'):
        model.optimizer.learning_rate.assign(best_lr)
    else:
        model.optimizer.learning_rate = best_lr

    # Plotting
    # plt.figure(figsize=(8, 5))
    # plt.plot(lrs, losses)
    # plt.xscale('log')
    # plt.xlabel("Learning Rate")
    # plt.ylabel("Loss")
    # plt.title("Learning Rate Finder")
    # plt.grid(True)
    # plt.savefig("lr_finder_plot.png")
    # plt.show()

def getEvenly(list_x, list_y):
    # 1. Find the target count
    count_0 = list_y.count(0)
    count_1 = list_y.count(1)
    target_count = min(count_0, count_1)

    if target_count == 0: # Handle empty or single-class lists
        return [], []

    # 2. Iterate and build indices, respecting the limit
    final_indices = []
    current_count_0 = 0
    current_count_1 = 0

    for i, y_val in enumerate(list_y):
        if y_val == 0:
            if current_count_0 < target_count:
                final_indices.append(i)
                current_count_0 += 1
        elif y_val == 1:
            if current_count_1 < target_count:
                final_indices.append(i)
                current_count_1 += 1

    # 3. Rebuild your lists
    new_X = [list_x[i] for i in final_indices]
    new_Y = [list_y[i] for i in final_indices]
    return new_X, new_Y

def trainModel(batch_size,initial_lr,fc_units,filter_sizes,kernel_sizes,opt_func,num_epoch=10,
               isconcat=True, applyG4Stack=False, applyRC=False, includeUnpairProb=False):
    #TRAIN PARAMETERS
    lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='loss',
        factor=0.5,
        patience=2,
        verbose=1,
        min_lr=1e-6,
        min_delta= 1e-5
    )


    model=prepModel.build_model( (5,100) if includeUnpairProb else (4,100),initial_lr,opt_func,dense_units=fc_units,filter_sizes=filter_sizes,
                                kernel_sizes=kernel_sizes,is_concatenated=isconcat,
                                applyG4Stack=applyG4Stack,
                                applyRCConv=applyRC)


    print("model is created.")
    model.summary()
    if (not os.path.isfile("data.pkl")):
        testingChromosomes=("chr19","Chr19","19")
        validationChromosomes=("chr20","Chr20","20")

        dirList=[("/home/osman/PycharmProjects/TFproject/BG4.c",1.0,True),("/home/osman/PycharmProjects/TFproject/negcont.c",0.0,False)]
        # dirList=[("/home/osman/PycharmProjects/TFproject/BG4neg",0.0)]


        allX=np.empty(shape=(0,5 if includeUnpairProb else 4,100) , dtype=np.float32)
        allY=np.empty(shape=(0,)  , dtype=np.float32)
        allChr=[]
        for dirName,defY,useReverse in dirList:
            print("Preparing", dirName, "...")
            dataX,dataY,dataChr=prepData.PrepDir(dirName, defaultY=defY, maxFilesToProcess=-1, includeOtherStrand=useReverse, includeUnpairProb=includeUnpairProb)
            # print(data)
            print(dirName, "complete. Size:", dataY.shape[0])

            allX=np.concatenate((allX,dataX))
            allY=np.concatenate((allY,dataY))
            allChr+=dataChr
            print("current sizes: allX:",allX.shape[0],"allY:",allY.shape[0],"allChr:",len(allChr))

        print("final sizes: allX:", allX.shape[0], "allY:", allY.shape[0],"allChr:",len(allChr))
        combined = list(zip(allX, allY, allChr))
        print("combined sizes:", len(combined))
        testAll=[]
        trainAll=[]
        validAll=[]
        for datum in combined:
            if (datum[2] in testingChromosomes):
                testAll.append(datum)
            elif (datum[2] in validationChromosomes):
                validAll.append(datum)
            else:
                trainAll.append(datum)
        print("final sizes: train:", len(trainAll), "test:",len(testAll), "validation:",len(validAll))
        random.shuffle(testAll)
        random.shuffle(trainAll)
        random.shuffle(validAll)

        trainX, trainY, _ = zip(*trainAll)
        testX, testY, _ = zip(*testAll)
        validX, validY, _ = zip(*validAll)
        assert len(trainY)==len(trainX), "train sizes do not match"
        assert len(testX) == len(testY), "test sizes do not match"
        assert len(validX) == len(validY), "test sizes do not match"
        # print("Files analyzed. Training size:",len(trainY),"Testing size:",len(testY))

        pkl.dump((trainX,trainY,testX,testY,validX,validY),open("data.pkl","wb"))
    else:
        print("Loading data file data.pkl...")
        trainX, trainY, testX, testY, validX, validY = pkl.load(open("data.pkl","rb"))


    testX,testY=getEvenly(testX,testY)
    validX,validY=getEvenly(validX,validY)


    print("Converting back to tensor")
    with device('/cpu:0'):
        # x = tf.convert_to_tensor(x, np.float32)
        # y = tf.convert_to_tensor(y, np.float32)

        trainX=convert_to_tensor(np.array(trainX), np.float32)
        trainY=convert_to_tensor(np.array(trainY), np.float32)
        testX=convert_to_tensor( np.array(testX) , np.float32)
        testY=convert_to_tensor( np.array(testY) , np.float32)
        validX=convert_to_tensor( np.array(validX) , np.float32)
        validY=convert_to_tensor( np.array(validY) , np.float32)


    print("Training started")
    print(trainX.shape,trainY.shape)

    print("Training data:")
    print("Positive counts:", tf.math.count_nonzero(tf.equal(trainY, 1.0)))
    print("Negative counts:",tf.math.count_nonzero(tf.equal(trainY, 0.0)))

    print("Validation data:")
    print("Positive counts:", tf.math.count_nonzero(tf.equal(validY, 1.0)))
    print("Negative counts:",tf.math.count_nonzero(tf.equal(validY, 0.0)))

    print("Testing data:")
    print("Positive counts:", tf.math.count_nonzero(tf.equal(testY, 1.0)))
    print("Negative counts:",tf.math.count_nonzero(tf.equal(testY, 0.0)))


    reduce_lr = ReduceLROnPlateau(monitor='val_auc', mode='max',
                                                     factor=0.5, patience=2, min_lr=1e-6, verbose=1)
    early = EarlyStopping(monitor='val_auc', mode='max',
                                             patience=5, restore_best_weights=True)
    # model.fit(trainX,trainY,epochs=num_epoch, batch_size=batch_size, callbacks=[lr_schedule] , validation_data=(testX,testY) )
    model.fit(trainX,trainY,epochs=num_epoch, batch_size=batch_size, callbacks=[reduce_lr,early] , validation_data=(validX,validY) )



    print("Training complete. starting evaluation using test set")
    evaluation=model.evaluate(x=testX,y=testY)
    print("Accuracy and loss:", evaluation)

    #auc,fpr,tpr,thr=plot_roc_curve(testX,testY,model,f"roc_curve_{auc}.png")
    # auc, fpr, tpr, thr = calculate_auc(testX, testY, model)
    roc_auc, fpr, tpr, thr, prec, recal, pr_thr =plot_roc_pr_curves(testX, testY, model,
                        filenamebase=f"roc_pr_curve")


    with open(f"roc_pr_curve_{roc_auc:.3f}.csv","w") as f:
        f.write("sep=,\n")
        f.write(f"batches,{batch_size},dense counts,{fc_units},filter sizes,"
                f"{filter_sizes},kernel sizes,{kernel_sizes},opt_func,{opt_func},"
                f"concat,{isconcat},G4Stack,{applyG4Stack},RC,{applyRC}\n")
        for label,data in [("tpr",tpr),("fpr",fpr),("ROC thresholds",thr),("precision",prec),("recall",recal),("PR thresholds",pr_thr)]:
            f.write(label)
            f.write(",")
            f.write(",".join(str(x) for x in data))
            f.write("\n")

    print("Area under curve:",roc_auc)
    print("Plot is saved as png")


    with open("hyperParameterResults.txt", "a") as f:
        f.write(f"auc: {roc_auc}\teval: {evaluation}\tbatch size: {batch_size}\tlr: {initial_lr}\tfc: {fc_units}\tfilters: {filter_sizes}\tkernels: {kernel_sizes}\topt func: {opt_func}\tepoch: {num_epoch}\tG4Stack: {applyG4Stack}\tRCconv: {applyRC}\n")

    model.save(f"bestmodel_{roc_auc}.keras")

    return model

if __name__ == "__main__":

    # import tensorflow as tf
    # gpus = tf.config.list_physical_devices('GPU')
    # if gpus:
    #     try:
    #         # Currently, memory growth needs to be the same across GPUs
    #         for gpu in gpus:
    #             tf.config.experimental.set_memory_growth(gpu, True)
    #         print("Success: GPU memory growth enabled.")
    #     except RuntimeError as e:
    #         # Memory growth must be set before GPUs have been initialized
    #         print(e)
    # trainModel(128, 0.001,(128, 64, 32), (64, 80, 128), (7, 9, 13), "Adam", 10)

    batches=[128]
    # lrs = [1e-2, 1e-3, 1e-4]
    # initial_lrs = [1e-3]
    # initial_lrs = np.geomspace(1e-05, 1e-1, 10)
    initial_lrs = [0.0006]

    # fcs=[(128, 64, 32)]
    # fcs=[(16, 32, 64),(32, 64, 128)]
    # fcs=[(32, 64, 128)] # this was best
    fcs=[(32, 64, 128),(128,64,32),(64,32),(32,16),(64,32,16),(32,),(64,)]
    fcs=[(32,),(32,16),(64,32),(64,)]
    fcs=[(64,32)]


    # filters = [(64, 80, 128)]
    # filters=[(64,)]
    # filters=[(64,80)] # this was best
    filters=[(64,80),(32,64),(16,40)]
    filters=[(16,40)]

    # kernels=[(11,7,5),(5,7,11)]
    # kernels=[(9,13,17),(7,9,13),(5,7,9),(5,7,11)]
    # kernels=[(5,7,9), (5,9,13)]*4
    # kernels=[(5,9,15)]
    # kernels=[(5,7,9)]
    # kernels=[(5,)]# this was best
    kernels=[(5,7),(5,9),(5,13),(7,5),(9,5),(13,5)]
    # kernels=[(3,7)]
    # kernels=[(5,7)]
    kernels=[(5,13)]

    # epochs=(10,15,20,25)
    epochs=(20,)

    g4=(False,True)*4
    rc=(False,True)*4
    #
    # g4=(False,)
    # rc=(False,)

    # g4=(True,)
    # rc=(True,)

    for b in batches:
        for li in initial_lrs:
            for fc in fcs:
                for f in filters:
                    for k in kernels:
                        for e in epochs:
                            for g in g4:
                                for r in rc:
                                    mdl=trainModel(b,li,fc,f,k, "Adam",e,applyG4Stack=g,applyRC=r)

    # fcs=[(32, 64, 128)]    filters=[(64,80)]  kernels=[(5,7)] Accuracy and loss: [0.1997826248407364, 0.9927306771278381, 0.9979091286659241]
