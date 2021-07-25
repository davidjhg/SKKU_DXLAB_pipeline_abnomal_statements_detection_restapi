import tensorflow as tf
import os
import numpy as np
import sklearn
import tensorflow.keras.backend as K
from tensorflow import keras
from PIL import Image
import numpy as np


config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.80
tf.keras.backend.set_session(tf.Session(config=config))



def my_iou_metric(label, pred):
    # Tensorflow version
    return tf.py_func(get_iou_vector, [label, pred > 0.5], tf.float64)


def mean_iou(y_true, y_pred):
    y_pred = K.cast(K.greater(y_pred, .5), dtype='float32') # .5 is the threshold
    inter = K.sum(K.sum(K.squeeze(y_true * y_pred, axis=3), axis=2), axis=1)
    union = K.sum(K.sum(K.squeeze(y_true + y_pred, axis=3), axis=2), axis=1) - inter
    return K.mean((inter + K.epsilon()) / (union + K.epsilon()))



def tf_mean_iou(y_true,y_pred):
    prec = []    
    y_pred_ = tf.to_int32(y_pred > 0.5)
    score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
    K.get_session().run(tf.local_variables_initializer())
    with tf.control_dependencies([up_opt]):
        score = tf.identity(score)
    return score


def iou_loss(y_true, y_pred):
    """Get Intersection over Union(IoU) ratio from ground truth and predicted masks.
    Arguments:
        y_true -- ground truth mask
        
        y_pred -- predicted mask
    """
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true * y_pred)
    score = (intersection + 1.) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection + 1.)
    return 1 - score

# mean iou as a metric
# https://www.kaggle.com/milaindl/unet-resnet50-classify-v1-0
def new_mean_iou(y_true, y_pred):
    """Get mean Intersection over Union(IoU) ratio from ground truth and predicted masks.
    Arguments:
        y_true -- ground truth mask
        
        y_pred -- predicted mask
    """
    y_pred = tf.round(y_pred)    
    intersect = tf.reduce_sum(y_true * y_pred, axis=[1]) 
    union = tf.reduce_sum(y_true, axis=[1]) + tf.reduce_sum(y_pred, axis=[1])
    smooth = tf.ones(tf.shape(intersect))
    return tf.reduce_mean((intersect + smooth) / (union - intersect + smooth))

def test_mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)


class MyMeanIOU(tf.keras.metrics.MeanIoU):
    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(tf.argmax(y_true, axis=-1), tf.argmax(y_pred, axis=-1), sample_weight)

# update
def iou_coef(y_true, y_pred, smooth=1):
    y_pred = K.cast(K.greater(y_pred, .5), dtype='float32')
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
    union = K.sum(y_true,[1,2,3])+K.sum(y_pred,[1,2,3])-intersection
    iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
    return iou

def dice_coef(y_true, y_pred, smooth=1):
    y_pred = K.cast(K.greater(y_pred, .5), dtype='float32')
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    dice = K.mean((2. * intersection + smooth)/(union + smooth), axis=0)
    return dice



def dsc(y_true, y_pred):
    smooth = 1.
    y_true_f = tf.keras.layers.Flatten()(y_true)
    y_pred_f = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    return score

def dice_loss(y_true, y_pred):
    loss = 1 - dsc(y_true, y_pred)
    return loss

def bce_dice_loss(y_true, y_pred):
    loss = tf.keras.losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss



# NOTE: ResUNet for crack detection 


def bn_act(x, act=True):
    x = keras.layers.BatchNormalization()(x)
    if act == True:
        x = keras.layers.Activation("relu")(x)
    return x

def conv_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    conv = bn_act(x)
    conv = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides)(conv)
    return conv

def stem(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    conv = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides)(x)
    conv = conv_block(conv, filters, kernel_size=kernel_size, padding=padding, strides=strides)
    
    shortcut = keras.layers.Conv2D(filters, kernel_size=(1, 1), padding=padding, strides=strides)(x)
    shortcut = bn_act(shortcut, act=False)
    
    output = keras.layers.Add()([conv, shortcut])
    return output

def residual_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    res = conv_block(x, filters, kernel_size=kernel_size, padding=padding, strides=strides)
    res = conv_block(res, filters, kernel_size=kernel_size, padding=padding, strides=1)
    
    shortcut = keras.layers.Conv2D(filters, kernel_size=(1, 1), padding=padding, strides=strides)(x)
    shortcut = bn_act(shortcut, act=False)
    
    output = keras.layers.Add()([shortcut, res])
    return output

def upsample_concat_block(x, xskip):
    u = keras.layers.UpSampling2D((2, 2))(x)
    c = keras.layers.Concatenate()([u, xskip])
    return c

def ResUNet(width:int,height:int):
    f = [16, 32, 64, 128, 256]
    inputs = keras.layers.Input((width,height, 3))
    
    ## Encoder
    e0 = inputs
    e1 = stem(e0, f[0])
    e2 = residual_block(e1, f[1], strides=2)
    e3 = residual_block(e2, f[2], strides=2)
    e4 = residual_block(e3, f[3], strides=2)
    e5 = residual_block(e4, f[4], strides=2)
    
    ## Bridge
    b0 = conv_block(e5, f[4], strides=1)
    b1 = conv_block(b0, f[4], strides=1)
    
    ## Decoder
    u1 = upsample_concat_block(b1, e4)
    d1 = residual_block(u1, f[4])
    
    u2 = upsample_concat_block(d1, e3)
    d2 = residual_block(u2, f[3])
    
    u3 = upsample_concat_block(d2, e2)
    d3 = residual_block(u3, f[2])
    
    u4 = upsample_concat_block(d3, e1)
    d4 = residual_block(u4, f[1])
    
    outputs = keras.layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid")(d4)
    model = keras.models.Model(inputs, outputs)
    
    return model

# NOTE: VGGUNet for crack detection 
def VGGFCN(width:int,height:int):
    inputs = tf.keras.layers.Input((width,height,3))
    ## 224 224 3
    
    flow = tf.keras.layers.Conv2D(64,(3,3),activation='relu',padding='same')(inputs)
    flow = tf.keras.layers.Conv2D(64,(3,3),activation='relu',padding='same')(flow)
    flow = tf.keras.layers.MaxPool2D((2,2))(flow)  
    
    pool1 = flow  ## 112 112 64
    
    flow = tf.keras.layers.Conv2D(128,(3,3),activation='relu',padding='same')(flow)
    flow = tf.keras.layers.Conv2D(128,(3,3),activation='relu',padding='same')(flow)
    flow = tf.keras.layers.MaxPool2D((2,2))(flow)  
    
    pool2 = flow  ## 56 56 128
    
    flow = tf.keras.layers.Conv2D(256,(3,3),activation='relu',padding='same')(flow)
    flow = tf.keras.layers.Conv2D(256,(3,3),activation='relu',padding='same')(flow)
    flow = tf.keras.layers.MaxPool2D((2,2))(flow)
    
    ## 28 28 256
    
    flow = tf.keras.layers.Conv2D(256,(1,1),activation='relu',padding='same')(flow)
    
    flow = tf.keras.layers.UpSampling2D((2,2))(flow)
    
    ## 56 56 256 
    
    flow = tf.keras.layers.Concatenate()([flow, pool2])
    
    flow = tf.keras.layers.Conv2DTranspose(256,(3,3),activation='relu',padding='same')(flow)
    flow = tf.keras.layers.Conv2DTranspose(256,(3,3),activation='relu',padding='same')(flow)
    flow = tf.keras.layers.Conv2DTranspose(256,(3,3),activation='relu',padding='same')(flow)
    
    flow = tf.keras.layers.UpSampling2D((2,2))(flow)
    
    flow = tf.keras.layers.Concatenate()([flow, pool1])
    
    flow = tf.keras.layers.Conv2D(128,(3,3),activation='relu',padding='same')(flow)
    flow = tf.keras.layers.Conv2D(128,(3,3),activation='relu',padding='same')(flow)
    
    flow = tf.keras.layers.UpSampling2D((2,2))(flow)
    flow = tf.keras.layers.Conv2D(64,(3,3),activation='relu',padding='same')(flow)
    flow = tf.keras.layers.Conv2D(64,(3,3),activation='relu',padding='same')(flow)
    
    output = tf.keras.layers.Conv2D(1,(1,1),activation='sigmoid',padding='same')(flow)
    
    model = tf.keras.models.Model(inputs,output)
    
    return model

crackRootDir = './9_crack_segmentation_road_224resized/'

def loadCrackImage(filename):
    original_img = Image.open(crackRootDir+'images/'+filename)
    img = (np.array(original_img).astype(np.float32)/255.0).reshape((1,224,224,3))
    #msk = (np.array(Image.open(crackRootDir+'masks/'+filename))/255.0).reshape(224,224,1).astype(np.int32).reshape((1,224,224,1))
    msk = Image.open(crackRootDir+'masks/'+filename)
    return img,msk,original_img

def TF2PIL(arr):
    return Image.fromarray(((arr.reshape((224,224))>=0.3)*255).astype(np.uint8))

def predictTF(model,img):
    return model.predict(img)



# NOTE: thermal image 

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F


class Stem(torch.nn.Module):
    def __init__(self,in_channels=3,out_channels=32):
        super().__init__()
        self.layer = torch.nn.Sequential(*[
            torch.nn.Conv2d(in_channels=in_channels,
                                      out_channels=out_channels,
                                     kernel_size=(3,3),stride=1,padding=1
                                 ),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(in_channels=out_channels,
                                      out_channels=out_channels,
                                     kernel_size=(3,3),stride=1,padding=1
                                 )
        ])
        
        self.shortcut = torch.nn.Sequential(*[
            torch.nn.Conv2d(in_channels=in_channels,
                           out_channels=out_channels,
                           kernel_size=(1,1),
                           stride=1),
            torch.nn.BatchNorm2d(out_channels)
        ])
        
    def forward(self,x):
        shortcut = self.shortcut(x)
        x = self.layer(x)
        return x + shortcut
        

class ResConvBlock(torch.nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.layer = torch.nn.Sequential(*[
            torch.nn.Conv2d(in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=(3,3),stride=1,padding=1
                                 ),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(in_channels=out_channels,
                            out_channels=out_channels,
                            kernel_size=(3,3),stride=1,padding=1
                                 ),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(True),
        ])
        self.shortcut = torch.nn.Sequential(*[
            torch.nn.Conv2d(in_channels=in_channels,
                           out_channels=out_channels,
                           kernel_size=(1,1),stride=1),
            torch.nn.BatchNorm2d(out_channels)
        ])
    def forward(self,x):
        shortcut = self.shortcut(x)
        x = self.layer(x)
        return x + shortcut
    


class IntertwinedUNet(torch.nn.Module):
    def __init__(self, n_class):
        super().__init__()
        self.n_features = [32,64,128,256,512]
        self.downsample = torch.nn.MaxPool2d(2)
        self.upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
                
        self.rgbStem = Stem(3,self.n_features[0])
        self.thrStem = Stem(3,self.n_features[0])
        
        self.rgbLayer1 = ResConvBlock(self.n_features[0],self.n_features[1])
        self.thrLayer1 = ResConvBlock(self.n_features[0],self.n_features[1])
        
        self.rgbLayer2 = ResConvBlock(self.n_features[1],self.n_features[2])
        self.thrLayer2 = ResConvBlock(self.n_features[1],self.n_features[2])
        
        self.rgbLayer3 = ResConvBlock(self.n_features[2],self.n_features[3])
        self.thrLayer3 = ResConvBlock(self.n_features[2],self.n_features[3])
        
        self.rgbLayer4 = ResConvBlock(self.n_features[3],self.n_features[4])
        self.thrLayer4 = ResConvBlock(self.n_features[3],self.n_features[4])
        
        self.bridge = ResConvBlock(self.n_features[4],self.n_features[4])
        
        self.dLayer4 = ResConvBlock(self.n_features[4]+self.n_features[3],self.n_features[3])
        self.dLayer3 = ResConvBlock(self.n_features[3]+self.n_features[2],self.n_features[2])
        self.dLayer2 = ResConvBlock(self.n_features[2]+self.n_features[1],self.n_features[1])
        self.dLayer1 = ResConvBlock(self.n_features[1]+self.n_features[0],self.n_features[0])
        
        self.out = torch.nn.Conv2d(self.n_features[0],n_class,1)
        
    def forward(self, rgb, thr):
        rgbStem = self.rgbStem(rgb)  # 256 
        
        rgb = self.downsample(rgbStem) # 128
        rgbLayer1 = self.rgbLayer1(rgb)
        
        rgb = self.downsample(rgbLayer1) # 64 
        rgbLayer2 = self.rgbLayer2(rgb)
        
        rgb = self.downsample(rgbLayer2) # 32 
        rgbLayer3 = self.rgbLayer3(rgb)
        
        rgb = self.downsample(rgbLayer3) # 16
        rgbLayer4 = self.rgbLayer4(rgb)
        
        
        thrStem = self.thrStem(thr)
        
        thr = self.downsample(thrStem)
        thrLayer1 = self.thrLayer1(thr)
        
        thr = self.downsample(thrLayer1)
        thrLayer2 = self.thrLayer2(thr)
        
        thr = self.downsample(thrLayer2)
        thrLayer3 = self.thrLayer3(thr)
        
        thr = self.downsample(thrLayer3)
        thrLayer4 = self.thrLayer4(thr)
        
        sumStem = rgbStem + thrStem  # 
        sumLayer1 = rgbLayer1 + thrLayer1  # 
        sumLayer2 = rgbLayer2 + thrLayer2  # 
        sumLayer3 = rgbLayer3 + thrLayer3  # 
        sumLayer4 = rgbLayer4 + thrLayer4  # 
        
        x = self.upsample(sumLayer4)
        x = torch.cat([x,sumLayer3],dim=1)
        x = self.dLayer4(x)
        
        x = self.upsample(x)
        x = torch.cat([x,sumLayer2],dim=1)
        x = self.dLayer3(x)
        
        x = self.upsample(x)
        x = torch.cat([x,sumLayer1],dim=1)
        x = self.dLayer2(x)
        
        x = self.upsample(x)
        x = torch.cat([x,sumStem],dim=1)
        x = self.dLayer1(x)
        
        out = self.out(x)
        
        return out
        


from torchvision import transforms
from PIL import Image

trans = transforms.Compose([
    transforms.ToTensor(),
])      


def loadRgbThermalImage(filename:str):
    rgbFileName = filename
    last = filename.split('/')[-1]
    thrFileName = rgbFileName[:-len(last)]+last.replace('RGB','THR')
    labFileName = rgbFileName[:-len(last)]+last.replace('RGB','LAB')
    rgb,thr,lab = Image.open(rgbFileName), Image.open(thrFileName), Image.open(labFileName).convert('L')
    rgbTensor,thrTensor,labTensor = trans(rgb),trans(thr),trans(lab)
    return rgbTensor.unsqueeze(0),thrTensor.unsqueeze(0),labTensor.unsqueeze(0),rgb,thr,lab

def predictTorch(model,rgb,thr):
    return model(rgb,thr).detach()

def TorchToPIL(pred):
    return Image.fromarray(((pred.squeeze().numpy()>=.5)*255).astype(np.uint8))

