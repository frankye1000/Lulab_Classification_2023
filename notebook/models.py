from multiprocessing import pool
import tensorflow as tf
from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2B0, EfficientNetV2B3

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

from config import efficientNet_config


def efficientNetV2B0_model():
    input_shape  = efficientNet_config['input_shape_B0']
    efnv2b0 = EfficientNetV2B0(include_top=False,
                               weights='imagenet', 
                               input_shape=input_shape,
                               pooling='avg'
                            )
    
    model = Sequential()
    model.add(efnv2b0)
    # model.add(GlobalAveragePooling2D())
    # model.add(Dropout(dropout_rate, name="dropout_out"))
    model.add(Dense(1, activation='sigmoid')) 
    # 輸出網絡模型參數
    model.summary() 
    # dot_img_file = 'EfficientNetV2B0.png'
    # tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)

    # 卷基層參與訓練
    efnv2b0.trainable = True 
    
    return model


def efficientNetV2B3_model():
    input_shape  = efficientNet_config['input_shape_B3']
    efnv2b3 = EfficientNetV2B3(include_top=False,
                               weights='imagenet', 
                               input_shape=input_shape,
                               pooling='avg'
                            )

    model = Sequential()
    model.add(efnv2b3)
    # model.add(GlobalAveragePooling2D())
    # model.add(Dropout(dropout_rate, name="dropout_out"))
    model.add(Dense(1, activation='sigmoid')) 
    # 輸出網絡模型參數
    model.summary() 

    # 卷基層參與訓練
    efnv2b3.trainable = True # True:全部解凍(超參數會很多)
    
    return model