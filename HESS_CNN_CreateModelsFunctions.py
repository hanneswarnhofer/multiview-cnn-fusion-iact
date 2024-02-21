import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras import layers, models, callbacks, optimizers, regularizers
from tensorflow.keras.layers import Input, Concatenate, concatenate, Dense,Lambda,Reshape,Embedding, Conv2D, Conv3D, MaxPooling2D,GlobalMaxPooling2D, MaxPooling3D, Flatten, Dropout, ConvLSTM2D, BatchNormalization, LeakyReLU, Maximum,GlobalMaxPooling1D
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint, LambdaCallback
from tensorflow.keras.models import Model, Sequential
from tensorflow import math
from keras import backend as K
from keras.regularizers import l2
from keras.losses import BinaryCrossentropy


from keras.layers import Input, Conv2D, BatchNormalization, Activation, Add, GlobalAveragePooling2D, Dense, AveragePooling2D
from keras.models import Model

input_shape = (41, 41, 1)
#dropout_rate = 0.2



#Define the model for the single-view CNNs
def create_base_model_moda(input_shape,kernel_size,dropout_rate,reg,pool_size,freeze=False):
    def build_model(inputs):
        Conv1 = Conv2D(filters=25, kernel_size=kernel_size, padding='same',kernel_regularizer=regularizers.l2(reg), input_shape=input_shape,)(inputs)
        LeakyRelu1 = LeakyReLU(alpha=0.1)(Conv1)
        MaxPool1 = MaxPooling2D(pool_size=pool_size, padding='same')(LeakyRelu1)

        Dropout1 = Dropout(dropout_rate)(MaxPool1)
        Conv2 = Conv2D(filters=30, kernel_size=kernel_size,padding='same', kernel_regularizer=regularizers.l2(reg))(Dropout1)
        LeakyRelu2 = LeakyReLU(alpha=0.1)(Conv2) 
        MaxPool2 = MaxPooling2D(pool_size=pool_size, padding='same')(LeakyRelu2)

        Dropout2 = Dropout(dropout_rate)(MaxPool2)
        Conv3 = Conv2D(filters=50, kernel_size=kernel_size,padding='same', kernel_regularizer=regularizers.l2(reg))(Dropout2)
        LeakyRelu3 = LeakyReLU(alpha=0.1)(Conv3) 
        MaxPool3 = MaxPooling2D(pool_size=pool_size, padding='same')(LeakyRelu3)

        Dropout3 = Dropout(dropout_rate)(MaxPool3)
        Conv4 = Conv2D(filters=50, kernel_size=kernel_size,padding='same', kernel_regularizer=regularizers.l2(reg))(Dropout3)
        LeakyRelu4 = LeakyReLU(alpha=0.1)(Conv4) 
        MaxPool4 = MaxPooling2D(pool_size=pool_size, padding='same')(LeakyRelu4)

        #### Split here for Early Fusion

        Dropout4 = Dropout(dropout_rate)(MaxPool4)
        Conv5 = Conv2D(filters=100, kernel_size=kernel_size,padding='same', kernel_regularizer=regularizers.l2(reg))(Dropout4)
        LeakyRelu5 = LeakyReLU(alpha=0.1)(Conv5) 
        MaxPool5 = MaxPooling2D(pool_size=pool_size, padding='same')(LeakyRelu5)

        Dropout5 = Dropout(dropout_rate)(MaxPool5)
        Conv6 = Conv2D(filters=100, kernel_size=kernel_size,padding='same', kernel_regularizer=regularizers.l2(reg))(Dropout5)
        MaxPool6 = MaxPooling2D(pool_size=pool_size, padding='same')(Conv6)

        Dropout6 = Dropout(dropout_rate)(MaxPool6)
        Conv7 = Conv2D(filters=200, kernel_size=kernel_size,padding='same', kernel_regularizer=regularizers.l2(reg))(Dropout6)
        MaxPool7 = MaxPooling2D(pool_size=pool_size, padding='same')(Conv7)

        Flat = Flatten()(MaxPool7)
        Dense1 = Dense(units=1024, activation='relu')(Flat)

        model = Model(inputs=inputs, outputs=Dense1)

        if freeze:
            for layer in model.layers:
                layer.trainable = False

        return model
    return build_model

def create_base_model_early(input_shape,freeze=False):
    dropout_rate = 0.2
    pool_size = 2
    kernel_size = 4
    reg = 0.00001
    model = Sequential()

    model.add(Conv2D(filters=25, kernel_size=kernel_size, activation='relu', padding='same',kernel_regularizer=regularizers.l2(reg), input_shape=input_shape,))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=pool_size, padding='same'))

    model.add(Dropout(dropout_rate))
    model.add(Conv2D(filters=30, kernel_size=kernel_size, activation='relu', padding='same', kernel_regularizer=regularizers.l2(reg)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=pool_size, padding='same'))

    model.add(Dropout(dropout_rate))
    model.add(Conv2D(filters=50, kernel_size=kernel_size, activation='relu', padding='same', kernel_regularizer=regularizers.l2(reg)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=pool_size, padding='same'))

    model.add(Dropout(dropout_rate))
    model.add(Conv2D(filters=50, kernel_size=kernel_size, activation='relu', padding='same',kernel_regularizer=regularizers.l2(reg)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=pool_size, padding='same'))


    if freeze:
        for layer in model.layers:
            layer.trainable = False

    return model

def create_base_model_early2(input_shape,freeze=False):
    dropout_rate = 0.2
    pool_size = 2
    kernel_size = 4
    reg = 0.00001
    model = Sequential()

    model.add(Conv2D(filters=25, kernel_size=kernel_size, activation='relu', padding='same',kernel_regularizer=regularizers.l2(reg), input_shape=input_shape,))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=pool_size, padding='same'))

    model.add(Dropout(dropout_rate))
    model.add(Conv2D(filters=30, kernel_size=kernel_size, activation='relu', padding='same', kernel_regularizer=regularizers.l2(reg)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=pool_size, padding='same'))


    if freeze:
        for layer in model.layers:
            layer.trainable = False

    return model


# Define the model for the combination of the previous CNNs and the final CNN for classification

def create_single_model(model):
    inputs = model.input
    x = model.output
    x = Dropout(0.5)(x)
    #x = Flatten(x)
    outputs = Dense(units=1, activation='sigmoid')(x)
    model_single = Model(inputs,outputs)
    #print_layer_dimensions(model_single)
    return model_single


def create_score_model(models,inputs,fusiontype):
    outputs = [model.output for model in models]
    if fusiontype == 'scoresum':
        # Sum-score fusion
        fused_output = Lambda(lambda x: K.sum(x, axis=-1,keepdims=True)/4)(concatenate(outputs, axis=-1))
    elif fusiontype == 'scoreproduct':
        # Product-score fusion
        fused_output = Lambda(lambda x: K.prod(x, axis=-1,keepdims=True))(concatenate(outputs, axis=-1))
    elif fusiontype == 'scoremax':
        # Max-score fusion
        fused_output = Lambda(lambda x: K.max(x, axis=-1,keepdims=True))(concatenate(outputs, axis=-1))
    else:
        raise ValueError("Invalid fusion_type. Choose from 'sum', 'product', or 'max'.")

    #fused_output_reshaped = Reshape((1,))(fused_output)
    #output = Dense(units=1, activation='sigmoid')(fused_output) #no activation=linear

    fused_model = Model(inputs,fused_output)

    return fused_model  

def create_latefc_model(models,inputs):
    fusionlayer = concatenate([model.output for model in models],axis=-1)
    fusionlayer = Dense(units=256,activation='relu')(fusionlayer)
    x = Dropout(0.5)(fusionlayer)
    #x = Flatten(x)
    outputs = Dense(units=1, activation='sigmoid')(x)
    model_multi = Model(inputs,outputs)
    #print_layer_dimensions(model_multi)
    return model_multi



def create_latemax_model(models,inputs):
    fusionlayer = concatenate([model.output for model in models],axis=0)
    print(fusionlayer.shape)
    fusionlayer = Lambda(lambda x: tf.reduce_max(x, axis=0), output_shape=input_shape)([model.output for model in models])

    x = Dropout(0.5)(fusionlayer)
    #x = Flatten(x)
    outputs = Dense(units=1, activation='sigmoid')(x)
    model_multi = Model(inputs,outputs)
    #print_layer_dimensions(model_multi)
    return model_multi

def create_earlyconv_model(models,inputs,dropout_rate,kernel_size,pool_size,reg):

    #outputs = [model(inputs[i]) for i, model in enumerate(models)]
    fusionlayer = concatenate(models, axis=-1)
    fused_feature_map = Conv2D(filters=fusionlayer.shape[-1] // 4, kernel_size=(1, 1), activation='relu', padding='same', kernel_regularizer=regularizers.l2(reg))(fusionlayer)

    Dropout4 = Dropout(dropout_rate)(fused_feature_map)
    Conv5 = Conv2D(filters=100, kernel_size=kernel_size,padding='same', kernel_regularizer=regularizers.l2(reg))(Dropout4)
    LeakyRelu5 = LeakyReLU(alpha=0.1)(Conv5) 
    MaxPool5 = MaxPooling2D(pool_size=pool_size, padding='same')(LeakyRelu5)

    Dropout5 = Dropout(dropout_rate)(MaxPool5)
    Conv6 = Conv2D(filters=100, kernel_size=kernel_size,padding='same', kernel_regularizer=regularizers.l2(reg))(Dropout5)
    MaxPool6 = MaxPooling2D(pool_size=pool_size, padding='same')(Conv6)

    Dropout6 = Dropout(dropout_rate)(MaxPool6)
    Conv7 = Conv2D(filters=200, kernel_size=kernel_size,padding='same', kernel_regularizer=regularizers.l2(reg))(Dropout6)
    MaxPool7 = MaxPooling2D(pool_size=pool_size, padding='same')(Conv7)

    Flat = Flatten()(MaxPool7)
    Dense1 = Dense(units=1024, activation='relu')(Flat)
    #x = Flatten(x)
    outputs = Dense(units=1, activation='sigmoid')(Dense1)
    model_multi = Model(inputs,outputs)
    #print_layer_dimensions(model_multi)
    return model_multi

def create_earlyconv2_model(models,inputs,dropout_rate,kernel_size,pool_size,reg):

    #outputs = [model(inputs[i]) for i, model in enumerate(models)]
    fusionlayer = concatenate(models, axis=-1)
    fused_feature_map = Conv2D(filters=fusionlayer.shape[-1] // 4, kernel_size=(1, 1), activation='relu', padding='same', kernel_regularizer=regularizers.l2(reg))(fusionlayer)

    Dropout2 = Dropout(dropout_rate)(fused_feature_map)
    Conv3 = Conv2D(filters=50, kernel_size=kernel_size,padding='same', kernel_regularizer=regularizers.l2(reg))(Dropout2)
    LeakyRelu3 = LeakyReLU(alpha=0.1)(Conv3) 
    MaxPool3 = MaxPooling2D(pool_size=pool_size, padding='same')(LeakyRelu3)

    Dropout3 = Dropout(dropout_rate)(MaxPool3)
    Conv4 = Conv2D(filters=50, kernel_size=kernel_size,padding='same', kernel_regularizer=regularizers.l2(reg))(Dropout3)
    LeakyRelu4 = LeakyReLU(alpha=0.1)(Conv4) 
    MaxPool4 = MaxPooling2D(pool_size=pool_size, padding='same')(LeakyRelu4)

    Dropout4 = Dropout(dropout_rate)(MaxPool4)
    Conv5 = Conv2D(filters=100, kernel_size=kernel_size,padding='same', kernel_regularizer=regularizers.l2(reg))(Dropout4)
    LeakyRelu5 = LeakyReLU(alpha=0.1)(Conv5) 
    MaxPool5 = MaxPooling2D(pool_size=pool_size, padding='same')(LeakyRelu5)

    Dropout5 = Dropout(dropout_rate)(MaxPool5)
    Conv6 = Conv2D(filters=100, kernel_size=kernel_size,padding='same', kernel_regularizer=regularizers.l2(reg))(Dropout5)
    MaxPool6 = MaxPooling2D(pool_size=pool_size, padding='same')(Conv6)

    Dropout6 = Dropout(dropout_rate)(MaxPool6)
    Conv7 = Conv2D(filters=200, kernel_size=kernel_size,padding='same', kernel_regularizer=regularizers.l2(reg))(Dropout6)
    MaxPool7 = MaxPooling2D(pool_size=pool_size, padding='same')(Conv7)

    Flat = Flatten()(MaxPool7)
    Dense1 = Dense(units=1024, activation='relu')(Flat)
    #x = Flatten(x)
    outputs = Dense(units=1, activation='sigmoid')(Dense1)
    model_multi = Model(inputs,outputs)
    #print_layer_dimensions(model_multi)
    return model_multi


def create_earlymax_model(models,inputs,dropout_rate,kernel_size,pool_size,reg):

    #outputs = [model(inputs[i]) for i, model in enumerate(models)]
    fusionlayer = concatenate(models, axis=-1)
    fusionlayer = Reshape((fusionlayer.shape[1], fusionlayer.shape[2], fusionlayer.shape[3] // 4, 4))(fusionlayer)

    max_pooling_function = Lambda(lambda x: K.max(x, axis=-1, keepdims=False))
    fused_feature_map = max_pooling_function(fusionlayer) 

    Dropout4 = Dropout(dropout_rate)(fused_feature_map)
    Conv5 = Conv2D(filters=100, kernel_size=kernel_size,padding='same', kernel_regularizer=regularizers.l2(reg))(Dropout4)
    LeakyRelu5 = LeakyReLU(alpha=0.1)(Conv5) 
    MaxPool5 = MaxPooling2D(pool_size=pool_size, padding='same')(LeakyRelu5)

    Dropout5 = Dropout(dropout_rate)(MaxPool5)
    Conv6 = Conv2D(filters=100, kernel_size=kernel_size,padding='same', kernel_regularizer=regularizers.l2(reg))(Dropout5)
    MaxPool6 = MaxPooling2D(pool_size=pool_size, padding='same')(Conv6)

    Dropout6 = Dropout(dropout_rate)(MaxPool6)
    Conv7 = Conv2D(filters=200, kernel_size=kernel_size,padding='same', kernel_regularizer=regularizers.l2(reg))(Dropout6)
    MaxPool7 = MaxPooling2D(pool_size=pool_size, padding='same')(Conv7)

    Flat = Flatten()(MaxPool7)
    Dense1 = Dense(units=1024, activation='relu')(Flat)
    #x = Flatten(x)
    outputs = Dense(units=1, activation='sigmoid')(Dense1)
    model_multi = Model(inputs,outputs)
    #print_layer_dimensions(model_multi)
    return model_multi

def create_earlymax2_model(models,inputs,dropout_rate,kernel_size,pool_size,reg):

    #outputs = [model(inputs[i]) for i, model in enumerate(models)]
    fusionlayer = concatenate(models, axis=-1)
    fusionlayer = Reshape((fusionlayer.shape[1], fusionlayer.shape[2], fusionlayer.shape[3] // 4, 4))(fusionlayer)

    max_pooling_function = Lambda(lambda x: K.max(x, axis=-1, keepdims=False))
    fused_feature_map = max_pooling_function(fusionlayer) 

    Dropout2 = Dropout(dropout_rate)(fused_feature_map)
    Conv3 = Conv2D(filters=50, kernel_size=kernel_size,padding='same', kernel_regularizer=regularizers.l2(reg))(Dropout2)
    LeakyRelu3 = LeakyReLU(alpha=0.1)(Conv3) 
    MaxPool3 = MaxPooling2D(pool_size=pool_size, padding='same')(LeakyRelu3)

    Dropout3 = Dropout(dropout_rate)(MaxPool3)
    Conv4 = Conv2D(filters=50, kernel_size=kernel_size,padding='same', kernel_regularizer=regularizers.l2(reg))(Dropout3)
    LeakyRelu4 = LeakyReLU(alpha=0.1)(Conv4) 
    MaxPool4 = MaxPooling2D(pool_size=pool_size, padding='same')(LeakyRelu4)

    Dropout4 = Dropout(dropout_rate)(MaxPool4)
    Conv5 = Conv2D(filters=100, kernel_size=kernel_size,padding='same', kernel_regularizer=regularizers.l2(reg))(Dropout4)
    LeakyRelu5 = LeakyReLU(alpha=0.1)(Conv5) 
    MaxPool5 = MaxPooling2D(pool_size=pool_size, padding='same')(LeakyRelu5)

    Dropout5 = Dropout(dropout_rate)(MaxPool5)
    Conv6 = Conv2D(filters=100, kernel_size=kernel_size,padding='same', kernel_regularizer=regularizers.l2(reg))(Dropout5)
    MaxPool6 = MaxPooling2D(pool_size=pool_size, padding='same')(Conv6)

    Dropout6 = Dropout(dropout_rate)(MaxPool6)
    Conv7 = Conv2D(filters=200, kernel_size=kernel_size,padding='same', kernel_regularizer=regularizers.l2(reg))(Dropout6)
    MaxPool7 = MaxPooling2D(pool_size=pool_size, padding='same')(Conv7)

    Flat = Flatten()(MaxPool7)
    Dense1 = Dense(units=1024, activation='relu')(Flat)
    #x = Flatten(x)
    outputs = Dense(units=1, activation='sigmoid')(Dense1)
    model_multi = Model(inputs,outputs)
    #print_layer_dimensions(model_multi)
    return model_multi


def create_earlyconcat_model(models,inputs,dropout_rate):
    #dropout_rate = 0.2 
    pool_size = 2
    kernel_size = 4
    reg = 0.00001

    #outputs = [model(inputs[i]) for i, model in enumerate(models)]
    merged = concatenate(models)

    Dropout4 = Dropout(dropout_rate)(merged)
    Conv5 = Conv2D(filters=100, kernel_size=kernel_size,padding='same', kernel_regularizer=regularizers.l2(reg))(Dropout4)
    LeakyRelu5 = LeakyReLU(alpha=0.1)(Conv5) 
    MaxPool5 = MaxPooling2D(pool_size=pool_size, padding='same')(LeakyRelu5)

    Dropout5 = Dropout(dropout_rate)(MaxPool5)
    Conv6 = Conv2D(filters=100, kernel_size=kernel_size,padding='same', kernel_regularizer=regularizers.l2(reg))(Dropout5)
    MaxPool6 = MaxPooling2D(pool_size=pool_size, padding='same')(Conv6)

    Dropout6 = Dropout(dropout_rate)(MaxPool6)
    Conv7 = Conv2D(filters=200, kernel_size=kernel_size,padding='same', kernel_regularizer=regularizers.l2(reg))(Dropout6)
    MaxPool7 = MaxPooling2D(pool_size=pool_size, padding='same')(Conv7)

    Flat = Flatten()(MaxPool7)
    Dense1 = Dense(units=1024, activation='relu')(Flat)

    Dropout6 = Dropout(dropout_rate)(Dense1)
    dense_layer_merged3 = Dense(units=1, activation='sigmoid')(Dropout6)

    model = Model(inputs=inputs, outputs=dense_layer_merged3)
    return model


def create_earlyconcat2_model(models,inputs,dropout_rate):
    #dropout_rate = 0.2 
    pool_size = 2
    kernel_size = 4
    reg = 0.00001

    #outputs = [model(inputs[i]) for i, model in enumerate(models)]
    merged = concatenate(models)

    Dropout2 = Dropout(dropout_rate)(merged)
    Conv3 = Conv2D(filters=50, kernel_size=kernel_size,padding='same', kernel_regularizer=regularizers.l2(reg))(Dropout2)
    LeakyRelu3 = LeakyReLU(alpha=0.1)(Conv3) 
    MaxPool3 = MaxPooling2D(pool_size=pool_size, padding='same')(LeakyRelu3)

    Dropout3 = Dropout(dropout_rate)(MaxPool3)
    Conv4 = Conv2D(filters=50, kernel_size=kernel_size,padding='same', kernel_regularizer=regularizers.l2(reg))(Dropout3)
    LeakyRelu4 = LeakyReLU(alpha=0.1)(Conv4) 
    MaxPool4 = MaxPooling2D(pool_size=pool_size, padding='same')(LeakyRelu4)

    Dropout4 = Dropout(dropout_rate)(MaxPool4)
    Conv5 = Conv2D(filters=100, kernel_size=kernel_size,padding='same', kernel_regularizer=regularizers.l2(reg))(Dropout4)
    LeakyRelu5 = LeakyReLU(alpha=0.1)(Conv5) 
    MaxPool5 = MaxPooling2D(pool_size=pool_size, padding='same')(LeakyRelu5)

    Dropout5 = Dropout(dropout_rate)(MaxPool5)
    Conv6 = Conv2D(filters=100, kernel_size=kernel_size,padding='same', kernel_regularizer=regularizers.l2(reg))(Dropout5)
    MaxPool6 = MaxPooling2D(pool_size=pool_size, padding='same')(Conv6)

    Dropout6 = Dropout(dropout_rate)(MaxPool6)
    Conv7 = Conv2D(filters=200, kernel_size=kernel_size,padding='same', kernel_regularizer=regularizers.l2(reg))(Dropout6)
    MaxPool7 = MaxPooling2D(pool_size=pool_size, padding='same')(Conv7)

    Flat = Flatten()(MaxPool7)
    Dense1 = Dense(units=1024, activation='relu')(Flat)

    Dropout6 = Dropout(dropout_rate)(Dense1)
    dense_layer_merged3 = Dense(units=1, activation='sigmoid')(Dropout6)

    model = Model(inputs=inputs, outputs=dense_layer_merged3)
    return model


# Define the model for the combination of the previous CNNs and the final CNN for classification

def create_multi_model(base, transfer, fusiontype, input_shape, kernel_size, dropout_rate, reg, pool_size, filters_1,startblock,endblock):
    print("\n #####################   MULTI VIEW MODEL   #######################")
    print("###### ",base, " ##### ",fusiontype," ######")
    if transfer == 'yes':

        if fusiontype == 'earlymax' or fusiontype == 'earlyconcat' or fusiontype == 'earlyconv':

            input_1 = Input(shape=input_shape)
            cnn_model_1 = create_base_model_early(input_shape, freeze=True)(input_1)
            cnn_model_1.load_weights('single_cnn_weights_partial.h5', by_name=True)

            input_2 = Input(shape=input_shape)
            cnn_model_2 = create_base_model_early(input_shape, freeze=True)(input_2)
            cnn_model_2.load_weights('single_cnn_weights_partial.h5', by_name=True)

            input_3 = Input(shape=input_shape)
            cnn_model_3 = create_base_model_early(input_shape, freeze=True)(input_3)
            cnn_model_3.load_weights('single_cnn_weights_partial.h5', by_name=True)

            input_4 = Input(shape=input_shape)
            cnn_model_4 = create_base_model_early(input_shape, freeze=True)(input_4)
            cnn_model_4.load_weights('single_cnn_weights_partial.h5', by_name=True)

        else:

            input_1 = Input(shape=input_shape)
            cnn_model_1 = create_base_model_moda(input_shape,kernel_size,dropout_rate,reg,pool_size, freeze=True)(input_1)
            cnn_model_1.load_weights('single_cnn_weights_partial.h5', by_name=True)

            input_2 = Input(shape=input_shape)
            cnn_model_2 = create_base_model_moda(input_shape,kernel_size,dropout_rate,reg, pool_size, freeze=True)(input_2)
            cnn_model_2.load_weights('single_cnn_weights_partial.h5', by_name=True)

            input_3 = Input(shape=input_shape)
            cnn_model_3 = create_base_model_moda(input_shape,kernel_size,dropout_rate,reg,pool_size, freeze=True)(input_3)
            cnn_model_3.load_weights('single_cnn_weights_partial.h5', by_name=True)

            input_4 = Input(shape=input_shape)
            cnn_model_4 = create_base_model_moda(input_shape,kernel_size,dropout_rate,reg,pool_size, freeze=True)(input_4)
            cnn_model_4.load_weights('single_cnn_weights_partial.h5', by_name=True)


    elif transfer == 'no':
        
        if fusiontype == 'earlymax' or fusiontype == 'earlyconcat' or fusiontype == 'earlyconv':

            input_1 = Input(shape=input_shape)
            cnn_model_1 = create_base_model_early(input_shape)(input_1)

            input_2 = Input(shape=input_shape)
            cnn_model_2 = create_base_model_early(input_shape)(input_2)

            input_3 = Input(shape=input_shape)
            cnn_model_3 = create_base_model_early(input_shape)(input_3)

            input_4 = Input(shape=input_shape)
            cnn_model_4 = create_base_model_early(input_shape)(input_4)

        elif fusiontype == 'earlymax2' or fusiontype == 'earlyconv2' or fusiontype == 'earlyconcat2':

            input_1 = Input(shape=input_shape)
            cnn_model_1 = create_base_model_early2(input_shape)(input_1)

            input_2 = Input(shape=input_shape)
            cnn_model_2 = create_base_model_early2(input_shape)(input_2)

            input_3 = Input(shape=input_shape)
            cnn_model_3 = create_base_model_early2(input_shape)(input_3)

            input_4 = Input(shape=input_shape)
            cnn_model_4 = create_base_model_early2(input_shape)(input_4)

        elif fusiontype == 'earlymax3':

            input_1 = Input(shape=input_shape)
            cnn_model_1 = create_base_model_early3(input_shape)(input_1)

            input_2 = Input(shape=input_shape)
            cnn_model_2 = create_base_model_early3(input_shape)(input_2)

            input_3 = Input(shape=input_shape)
            cnn_model_3 = create_base_model_early3(input_shape)(input_3)

            input_4 = Input(shape=input_shape)
            cnn_model_4 = create_base_model_early3(input_shape)(input_4)

        else: 

            input_1 = Input(shape=input_shape)
            cnn_model_1 = create_base_model_moda(input_shape,kernel_size,dropout_rate,reg,pool_size)(input_1)

            input_2 = Input(shape=input_shape)
            cnn_model_2 = create_base_model_moda(input_shape,kernel_size,dropout_rate,reg,pool_size)(input_2)

            input_3 = Input(shape=input_shape)
            cnn_model_3 = create_base_model_moda(input_shape,kernel_size,dropout_rate,reg,pool_size)(input_3)

            input_4 = Input(shape=input_shape)
            cnn_model_4 = create_base_model_moda(input_shape,kernel_size,dropout_rate,reg,pool_size)(input_4)   

    #cnn2_model = create_CNN2_model_moda(input_shape,kernel_size,dropout_rate,reg,pool_size)
    if fusiontype == 'earlymax':
        model_multi = create_earlymax_model([cnn_model_1, cnn_model_2, cnn_model_3, cnn_model_4],[input_1, input_2, input_3, input_4],dropout_rate,kernel_size,pool_size,reg)
    elif fusiontype == 'earlymax2':
        model_multi = create_earlymax2_model([cnn_model_1, cnn_model_2, cnn_model_3, cnn_model_4],[input_1, input_2, input_3, input_4],dropout_rate,kernel_size,pool_size,reg)
    elif fusiontype == 'earlyconv':
        model_multi = create_earlyconv_model([cnn_model_1, cnn_model_2, cnn_model_3, cnn_model_4],[input_1, input_2, input_3, input_4],dropout_rate,kernel_size,pool_size,reg)
    elif fusiontype == 'earlyconv2':
        model_multi = create_earlyconv2_model([cnn_model_1, cnn_model_2, cnn_model_3, cnn_model_4],[input_1, input_2, input_3, input_4],dropout_rate,kernel_size,pool_size,reg)
    elif fusiontype == 'earlyconcat':
        model_multi = create_earlyconcat_model([cnn_model_1, cnn_model_2, cnn_model_3, cnn_model_4],[input_1, input_2, input_3, input_4],dropout_rate)  
    elif fusiontype == 'earlyconcat2':
        model_multi = create_earlyconcat2_model([cnn_model_1, cnn_model_2, cnn_model_3, cnn_model_4],[input_1, input_2, input_3, input_4],dropout_rate)  
    elif fusiontype == "latefc":
        model_multi = create_latefc_model([cnn_model_1, cnn_model_2, cnn_model_3, cnn_model_4],[input_1, input_2, input_3, input_4])
    elif fusiontype == "latemax":
        model_multi = create_latemax_model([cnn_model_1, cnn_model_2, cnn_model_3, cnn_model_4],[input_1, input_2, input_3, input_4])
    elif fusiontype == 'scoresum' or fusiontype == 'scoreproduct' or fusiontype == 'scoremax':
        single_view_models = [create_single_model(cnn_model_1), create_single_model(cnn_model_2), create_single_model(cnn_model_3), create_single_model(cnn_model_4)]
        model_multi = create_score_model(single_view_models,[input_1, input_2, input_3, input_4],fusiontype)
    else: print("ERROR: Fusiontype not known!!")

    return model_multi
