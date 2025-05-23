from tensorflow.keras import layers
from model_utils.layers import CONV_stack,encode_layer,decode_layer, attention_gate
def UNET_left(X, channel, kernel_size=3, stack_num=2, dropout_rate=0, activation='ReLU', 
              pool=True, batch_norm=False, name='left0'):
    '''
    The encoder block of U-net.
    
    UNET_left(X, channel, kernel_size=3, stack_num=2, activation='ReLU', 
              pool=True, batch_norm=False, name='left0')
    
    Input
    ----------
        X: input tensor.
        channel: number of convolution filters.
        kernel_size: size of 2-d convolution kernels.
        stack_num: number of convolutional layers.
        activation: one of the `tensorflow.keras.layers` interface, e.g., 'ReLU'.
        pool: True or 'max' for MaxPooling2D.
              'ave' for AveragePooling2D.
              False for strided conv + batch norm + activation.
        batch_norm: True for batch normalization, False otherwise.
        name: prefix of the created keras layers.
        
    Output
    ----------
        X: output tensor.
        
    '''
    pool_size = 2
    
    X = encode_layer(X, channel, pool_size, pool, activation=activation, 
                     batch_norm=batch_norm, name='{}_encode'.format(name))

    X = CONV_stack(X, channel, kernel_size,dropout_rate=dropout_rate, stack_num=stack_num, activation=activation, 
                   batch_norm=batch_norm, name='{}_conv'.format(name))
    
    return X


def UNET_right(X, X_list, channel,dropout_rate=0, kernel_size=3, 
               stack_num=2, activation='ReLU',
               unpool=True, batch_norm=False, concat=True, name='right0'):
    
    '''
    The decoder block of U-net.
    
    Input
    ----------
        X: input tensor.
        X_list: a list of other tensors that connected to the input tensor.
        channel: number of convolution filters.
        kernel_size: size of 2-d convolution kernels.
        stack_num: number of convolutional layers.
        activation: one of the `tensorflow.keras.layers` interface, e.g., 'ReLU'.
        unpool: True or 'bilinear' for Upsampling2D with bilinear interpolation.
                'nearest' for Upsampling2D with nearest interpolation.
                False for Conv2DTranspose + batch norm + activation.
        batch_norm: True for batch normalization, False otherwise.
        concat: True for concatenating the corresponded X_list elements.
        name: prefix of the created keras layers.
        
    Output
    ----------
        X: output tensor.
    
    '''
    
    pool_size = 2
    
    X = decode_layer(X, channel, pool_size, unpool, 
                     activation=activation, batch_norm=batch_norm, name='{}_decode'.format(name))
    
    # linear convolutional layers before concatenation
    X = CONV_stack(X, channel, kernel_size,dropout_rate=dropout_rate, stack_num=1, activation=activation, 
                   batch_norm=batch_norm, name='{}_conv_before_concat'.format(name))
    if concat:
        # <--- *stacked convolutional can be applied here
        X = layers.concatenate([X,]+X_list, axis=3, name=name+'_concat')
    
    # Stacked convolutions after concatenation 
    X = CONV_stack(X, channel, kernel_size,dropout_rate=dropout_rate, stack_num=stack_num, activation=activation, 
                   batch_norm=batch_norm, name=name+'_conv_after_concat')
    
    return X


def UNET_att_right(X, X_left, channel, att_channel,dropout_rate=0, kernel_size=3, stack_num=2,
                   activation='ReLU', atten_activation='ReLU', attention='add',
                   unpool=True, batch_norm=False, name='right0'):
    '''
    the decoder block of Attention U-net.
    
    UNET_att_right(X, X_left, channel, att_channel, kernel_size=3, stack_num=2,
                   activation='ReLU', atten_activation='ReLU', attention='add',
                   unpool=True, batch_norm=False, name='right0')
    
    Input
    ----------
        X: input tensor
        X_left: the output of corresponded downsampling output tensor (the input tensor is upsampling input)
        channel: number of convolution filters
        att_channel: number of intermediate channel.        
        kernel_size: size of 2-d convolution kernels.
        stack_num: number of convolutional layers.
        activation: one of the `tensorflow.keras.layers` or `keras_unet_collection.activations` interfaces, e.g., 'ReLU'.
        atten_activation: a nonlinear attnetion activation.
                    The `sigma_1` in Oktay et al. 2018. Default is 'ReLU'.
        attention: 'add' for additive attention. 'multiply' for multiplicative attention.
                   Oktay et al. 2018 applied additive attention.
        unpool: True or "bilinear" for Upsampling2D with bilinear interpolation.
                "nearest" for Upsampling2D with nearest interpolation.
                False for Conv2DTranspose + batch norm + activation.  
        batch_norm: True for batch normalization, False otherwise.
        name: prefix of the created keras layers.
    Output
    ----------
        X: output tensor.
    
    '''
    
    pool_size = 2
    
    X = decode_layer(X, channel, pool_size, unpool, 
                     activation=activation, batch_norm=batch_norm, name='{}_decode'.format(name))
    
    X_left = attention_gate(X=X_left, g=X, channel=att_channel, activation=atten_activation, 
                            attention=attention, name='{}_att'.format(name))
    
    # Tensor concatenation
    H = layers.concatenate([X, X_left], axis=-1, name='{}_concat'.format(name))
    
    # stacked linear convolutional layers after concatenation
    H = CONV_stack(H, channel, kernel_size,dropout_rate=dropout_rate, stack_num=stack_num, activation=activation, 
                   batch_norm=batch_norm, name='{}_conv_after_concat'.format(name))
    
    return H