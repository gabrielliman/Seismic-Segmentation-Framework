import argparse
import tensorflow as tf
import os
from models.attention import Attention_unet
from models.unet import Unet
from models.unet3plus import Unet_3plus
from models.bridgenet import FlexibleBridgeNet
from models.efficientNetB1 import EfficientNetB1
from focal_loss import SparseCategoricalFocalLoss
from utils.datapreparation import *
from utils.prediction import make_prediction
import matplotlib.pyplot as plt

from models.CFPNetM import CFPNetM
from models.ENet import ENet
from models.ESPNet import ESPNet
from models.ICNet import ICNet


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')

    parser.add_argument('--optimizer', '-o', dest='optimizer', metavar='O', type=int, default=0,
                        help="Choose optimizer, 0: Adam, 1: SGD, 2: RMS")

    parser.add_argument('--loss', '-l', dest='loss', metavar='L', type=int, default=0,
                        help="Choose loss function, 0= Cross Entropy, 1= Focal Loss")

    parser.add_argument('--gamma', '-g', dest='gamma', metavar='G', type=float, default=3.6,
                        help="Gamma for Sparse Categorical Focal Loss")

    parser.add_argument('--epochs', '-e', dest='epochs', metavar='E', type=int, default=100,
                        help='Limit of epochs')

    parser.add_argument('--patience', '-p', dest='patience', metavar='P', type=int, default=15,
                        help="Patience for callback function")

    parser.add_argument('--delta', '-d', dest='delta', metavar='D', type=float, default=1e-4,
                        help="Delta for callback function")

    parser.add_argument('--batch_size', '-b', dest='batch_size', metavar='B', type=int, default=16,
                        help='Batch size')

    parser.add_argument('--kernel', dest='kernel', metavar='K', type=int, default=7,
                        help="Kernel Size")

    parser.add_argument('--model', '-m', dest='model', metavar='M', type=int, default=0,
                        help="Choose Segmentation Model, 0: Unet, 1: Unet 3 Plus, 2: Attention UNet, 3: Flexible BridgeNet, 4: CFPNetM, 5: ENet, 6: ESPNet, 7: ICNet, 8: EfficientNetB1")

    parser.add_argument('--filters', dest='filters', metavar='F', type=int, default=6,
                        help="Number of filters for the model")

    parser.add_argument('--folder', '-f', dest='folder', metavar='FOLDER', type=str, default="default_folder",
                        help='Name of the folder where the results will be saved')

    parser.add_argument('--name', '-n', dest='name', metavar='NAME', type=str, default="default",
                        help='Model name for saving')

    parser.add_argument('--gpuID', dest='gpuID', metavar='GPU', type=int, default=1,
                        help="GPU id")

    parser.add_argument('--dataset', dest='dataset', metavar='DSET', type=int, default=0,
                        help="0: Parihaka, 1: Penobscot")

    parser.add_argument('--slice_height', '-s1', dest='slice_height', metavar='S1', type=int, default=992,
                        help='First dimension of image slices')

    parser.add_argument('--slice_width', '-s2', dest='slice_width', metavar='S2', type=int, default=192,
                        help='Second dimension of image slices')

    parser.add_argument('--train_val_stride_height', dest='train_val_stride_height', metavar='TSH', type=int, default=128,
                        help="Stride in first dimension for train images")

    parser.add_argument('--train_val_stride_width', dest='train_val_stride_width', metavar='TSW', type=int, default=64,
                        help="Stride in second dimension for train images")

    parser.add_argument('--test_stride_height', dest='test_stride_height', metavar='TeSH', type=int, default=128,
                        help="Stride in first dimension for test images")

    parser.add_argument('--test_stride_width', dest='test_stride_width', metavar='TeSW', type=int, default=64,
                        help="Stride in second dimension for test images")

    parser.add_argument('--train_limit_x', dest='train_limit_x', metavar='TLX', type=int, default=192,
                        help="Limit of x dimension of training for RPRV and RPEDS")

    parser.add_argument('--train_limit_y', dest='train_limit_y', metavar='TLY', type=int, default=192,
                        help="Limit of y dimension of training for RPRV and RPEDS")

    parser.add_argument('--extra_train_slices', dest='extra_train_slices', metavar='ETS', type=int, default=2,
                        help="Number of extra slices for RPEDS and EDS")

    return parser.parse_args()

if __name__ == '__main__':
    args= get_args()

    slice_height=args.slice_height
    slice_width=args.slice_width
    train_val_stride_height=args.train_val_stride_height
    train_val_stride_width=args.train_val_stride_width
    test_stride_height=args.test_stride_height
    test_stride_width=args.test_stride_width
    train_limit_x=args.train_limit_x
    train_limit_y=args.train_limit_y
    extra_train_slices=args.extra_train_slices

    gpuID=args.gpuID
    num_filters=int(args.filters)
    dataset_id=args.dataset
    model_id=args.model
    delta=args.delta
    patience=args.patience
    optimizer_id=args.optimizer
    loss_id=args.loss
    folder=args.folder
    name=args.name
    batch_size=args.batch_size
    epochs=args.epochs
    kernel=args.kernel
    gamma=args.gamma

    if(dataset_id==0):
        num_classes=6
        train_image,train_label, test_image, test_label, val_image, val_label=LRP_Parihaka(shape=(slice_height,slice_width), stridetrain=(train_val_stride_height,train_val_stride_width), strideval=(train_val_stride_height,train_val_stride_width), stridetest=(test_stride_height,test_stride_width))
    elif(dataset_id==1):
        num_classes=8
        train_image,train_label, test_image, test_label, val_image, val_label=LRP_Penobscot(shape=(slice_height,slice_width), stridetrain=(train_val_stride_height,train_val_stride_width), strideval=(train_val_stride_height,train_val_stride_width), stridetest=(test_stride_height,test_stride_width))
    elif(dataset_id==2):
        num_classes=6
        train_image,train_label, test_image, test_label, val_image, val_label=RPRV_Parihaka(shape=(slice_height,slice_width), stridetrain=(train_val_stride_height,train_val_stride_width), strideval=(train_val_stride_height,train_val_stride_width), stridetest=(test_stride_height,test_stride_width), sizetrain_x=train_limit_x, sizetrain_y=train_limit_y)
    elif (dataset_id==3):
        num_classes=8
        train_image,train_label, test_image, test_label, val_image, val_label=RPRV_Penobscot(shape=(slice_height,slice_width), stridetrain=(train_val_stride_height,train_val_stride_width), strideval=(train_val_stride_height,train_val_stride_width), stridetest=(test_stride_height,test_stride_width), sizetrain_x=train_limit_x, sizetrain_y=train_limit_y)
    elif(dataset_id==4):
        num_classes=6
        train_image,train_label, test_image, test_label, val_image, val_label=RPEDS_Parihaka(shape=(slice_height,slice_width), stridetrain=(train_val_stride_height,train_val_stride_width), strideval=(train_val_stride_height,train_val_stride_width), stridetest=(test_stride_height,test_stride_width), sizetrain_x=train_limit_x, sizetrain_y=train_limit_y, extra_train_slices=extra_train_slices)
    elif(dataset_id==5):
        num_classes=8
        train_image,train_label, test_image, test_label, val_image, val_label=RPEDS_Penobscot(shape=(slice_height,slice_width), stridetrain=(train_val_stride_height,train_val_stride_width), strideval=(train_val_stride_height,train_val_stride_width), stridetest=(test_stride_height,test_stride_width), sizetrain_x=train_limit_x, sizetrain_y=train_limit_y, extra_train_slices=extra_train_slices)
    elif(dataset_id==6):
        num_classes=6
        train_image,train_label, test_image, test_label, val_image, val_label=EDS_Parihaka(shape=(slice_height,slice_width), stridetrain=(train_val_stride_height,train_val_stride_width), strideval=(train_val_stride_height,train_val_stride_width), stridetest=(test_stride_height,test_stride_width), num_train=extra_train_slices)
    elif (dataset_id==7):
        num_classes=8
        train_image,train_label, test_image, test_label, val_image, val_label=EDS_Penobscot(shape=(slice_height,slice_width), stridetrain=(train_val_stride_height,train_val_stride_width), strideval=(train_val_stride_height,train_val_stride_width), stridetest=(test_stride_height,test_stride_width), num_train=extra_train_slices)

    if gpuID == -1:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        print('CPU is used.')
    elif gpuID == 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        print('GPU device ' + str(gpuID) + ' is used.')
    elif gpuID == 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        print('GPU device ' + str(gpuID) + ' is used.')

    filters=[]
    for i in range(0,num_filters):
        filters.append(2**(4 + i))

    if(model_id== 0):
        model = Unet(tam_entrada=(slice_height, slice_width, 1), num_filtros=filters, classes=num_classes,kernel_size=kernel)
    elif(model_id== 1):
        model = Unet_3plus(tam_entrada=(slice_height, slice_width, 1), n_filters=filters, classes=num_classes,kernel_size=kernel)
    elif(model_id== 2):
        model = Attention_unet(tam_entrada=(slice_height, slice_width, 1), num_filtros=filters, classes=num_classes,kernel_size=kernel)
    elif(model_id== 3):
        model = FlexibleBridgeNet(input_size=(slice_height,slice_width,1),up_down_times=5, Y_channels=num_classes, kernel_size=kernel,
                                kernels_all=[16, 32, 64, 128, 256, 512][0:6], conv2act_repeat=2, res_case=0,
                                res_number=0)
    elif(model_id== 4):
        model = CFPNetM(slice_height, slice_width, 1, num_classes)

    elif(model_id== 5):
        model = ENet(slice_height, slice_width, 1, num_classes)

    elif(model_id== 6):
        model = ESPNet(slice_height, slice_width, 1, num_classes)

    elif(model_id== 7):
        model = ICNet(slice_height, slice_width, 1, num_classes)

    elif(model_id== 8):
        model = EfficientNetB1(slice_height,slice_width, 1, num_classes)
        
    checkpoint_filepath = './checkpoints/' + folder + '/checkpoint_' + name  + '.weights.h5'

    if not os.path.exists('./checkpoints'):
        os.makedirs('./checkpoints')

    if not os.path.exists('./checkpoints/' + folder):
        os.makedirs('./checkpoints/' + folder)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            min_delta=delta,
            patience=patience,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            monitor= 'val_unet3plus_output_sup2_activation_acc' if model_id == 1 else "val_acc",
            mode='max',
            save_best_only=True
        )
    ]
    
    if(optimizer_id==0):
        opt=tf.keras.optimizers.Adam(learning_rate=1e-4)
        opt_name="Adam"
    elif(optimizer_id==1):
        opt=tf.keras.optimizers.SGD()
        opt_name="SGD"
    elif(optimizer_id==2):
        opt=tf.keras.optimizers.RMSprop()
        opt_name="RMS"

    if(loss_id==0):
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        loss_name="Sparce Categorical Cross Entropy"
    elif(loss_id==1):
        loss=SparseCategoricalFocalLoss(gamma=gamma, from_logits=True)
        loss_name="Sparce Categorical Focal Loss, Gamma: " + str(gamma)

    model.compile(optimizer=opt,
                        loss=loss,
                        metrics=['acc'])

    history = model.fit(train_image, train_label, batch_size=batch_size, epochs=epochs,
                            callbacks=callbacks,
                            validation_data=(val_image, val_label))     
    model.load_weights(checkpoint_filepath)

    if not os.path.exists('./results/' + folder):
        os.makedirs('./results/' + folder)

    if not os.path.exists('./results/' + folder + '/graphs'):
        os.makedirs('./results/' + folder + '/graphs')

    if not os.path.exists('./results/' + folder + '/tables'):
        os.makedirs('./results/' + folder + '/tables')

    if model_id==1:
        fig, axis = plt.subplots(1, 2, figsize=(20, 5))
        axis[0].plot(history.history["unet3plus_output_final_activation_loss"], color='r', label = 'train loss')
        axis[0].plot(history.history["val_unet3plus_output_final_activation_loss"], color='b', label = 'val loss')
        axis[0].set_title('Loss Comparison')
        axis[0].legend()
        axis[1].plot(history.history["unet3plus_output_final_activation_acc"], color='r', label = 'train acc')
        axis[1].plot(history.history["val_unet3plus_output_final_activation_acc"], color='b', label = 'val acc')
        axis[1].set_title('Accuracy Comparison')
        axis[1].legend()
        plt.grid(False)
    else:
        fig, axis = plt.subplots(1, 2, figsize=(20, 5))
        axis[0].plot(history.history["loss"], color='r', label='train loss')
        axis[0].plot(history.history["val_loss"], color='b', label='val loss')
        axis[0].set_title('Loss Comparison')
        axis[0].legend()
        axis[1].plot(history.history["acc"], color='r', label='train acc')
        axis[1].plot(history.history["val_acc"], color='b', label='val acc')
        axis[1].set_title('Accuracy Comparison')
        axis[1].legend()
        plt.grid(False)
    fig.savefig("results/" + folder + "/graphs/graph_" + name + ".png")

    # if not os.path.exists('./trained_models'):
    #     os.makedirs('./trained_models')
    # if not os.path.exists('./trained_models/' + folder):
    #     os.makedirs('./trained_models/' + folder)
    # model.save("./trained_models/" + folder + "_" + name + ".keras")

    make_prediction(name,folder,model, test_image, test_label, num_classes)
    f = open("results/" + folder + "/tables/table_" + name + ".txt", "a")
    model_info="\n\nModel: " + str(model.name) + "\nSlices: " +  str(slice_height) + "x" + str(slice_width) + "\nEpochs: " + str(epochs) + "\nDelta: " +  str(delta) + "\nPatience: " + str(patience) +  "\nBatch size: " + str(batch_size) + "\nOtimizador: "  + str(opt_name) + "\nFunção de Perda: " +  str(loss_name)
    f.write(model_info)
    stride_info="\n\nStride Train: " + str(train_val_stride_height) + "x" + str(train_val_stride_width) + "\nStride Validation: " + str(train_val_stride_height) + "x" + str(train_val_stride_width) + "\nStride Test: " + str(test_stride_height) + "x" + str(test_stride_width)
    f.write(stride_info)
    f.close()