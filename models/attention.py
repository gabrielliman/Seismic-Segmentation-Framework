from model_utils.attention_unet import my_att_unet_2d


def Attention_unet(tam_entrada, num_filtros, classes, dropout_rate=0, kernel_size=3):
    model = my_att_unet_2d(
        input_size=tam_entrada, filter_num=num_filtros, n_labels=classes,kernel_size=kernel_size, dropout_rate=dropout_rate,
        stack_num_down=5, stack_num_up=4, activation='ReLU',
        atten_activation='ReLU', attention='add', output_activation='Softmax',
        batch_norm=True, pool=False, unpool='bilinear', name='attunet'
    )

    model._name = "attunet"

    return model