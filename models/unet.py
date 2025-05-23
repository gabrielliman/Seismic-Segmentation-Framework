from model_utils.unet import my_unet_2d
def Unet(tam_entrada, num_filtros, classes, dropout_rate=0, kernel_size=3):
    model=my_unet_2d(
        input_size=tam_entrada, filter_num=num_filtros, n_labels=classes, kernel_size=kernel_size, dropout_rate=dropout_rate,
        stack_num_down=2, stack_num_up=2, activation="ReLU", output_activation="Softmax",
        batch_norm=True, pool="max", name='unet')

    model._name='unet'
    return model