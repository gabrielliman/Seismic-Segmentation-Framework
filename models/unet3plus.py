from model_utils.unet3plus import my_unet_3plus_2d
def Unet_3plus(tam_entrada, n_filters, classes, dropout_rate=0, kernel_size=3):

    model = my_unet_3plus_2d(
        input_size=tam_entrada, n_labels=classes, filter_num_down=n_filters,kernel_size=kernel_size, dropout_rate=dropout_rate,
        filter_num_skip='auto', filter_num_aggregate='auto',
        stack_num_down=2, stack_num_up=2, activation='ReLU', output_activation='Softmax',
        batch_norm=True, pool='max', unpool=False, deep_supervision=True, name='unet3plus'
    )

    model._name = "unet3plus"

    return model