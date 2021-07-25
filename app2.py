

import numpy as np 
import utils
import tensorflow as tf
import torch
from tensorflow import keras 
from torchvision import transforms
import torch
from PIL import Image

session = tf.Session()

keras.backend.set_session(session)


with tf.device('cpu:0'):
    vggUNet = utils.VGGFCN(224,224)
    resUNet = utils.ResUNet(224,224)
    vggUNet.load_weights('VGGFCN_KFOLD_0.h5')
    resUNet.load_weights('UNET_KFOLD0.h5')
    print('Crack Detection models are loaded')

crackList = np.load('./crackTestList.npy')
crackRootDir = './9_crack_segmentation_road_224resized/'

IntertUNet = utils.IntertwinedUNet(1)
IntertUNet.load_state_dict(torch.load('./IntertwinedUNet_cpu.pth'))
IntertUNet.eval()
print('Suspected Region detection model is loaded')

rgbFiles = np.load('./rgb_files.npy')

def getCrackPred(model,filename):
    img,msk,original_img = utils.loadCrackImage(filename)
    pred = utils.predictTF(model,img)
    pred_PIL = utils.TF2PIL(pred)
    return pred_PIL,original_img,msk

def getThermalPred(model,filename):
    rgb,thr,lab,r,t,l = utils.loadRgbThermalImage(filename)
    pred = utils.predictTorch(IntertUNet,rgb,thr)
    pred_PIL = utils.TorchToPIL(pred)
    return pred_PIL,r,t,l 

with tf.device('cpu:0'):
    print(crackList[0])
    print(getCrackPred(vggUNet,crackList[0]))
    print(getCrackPred(resUNet,crackList[1]))

from flask import Flask, jsonify, request

app = Flask(__name__)
print('[DEBUG] server is ready to receive')

def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    outputs = model.forward(tensor)
    _, y_hat = outputs.max(1)
    predicted_idx = str(y_hat.item())
    return imagenet_class_index[predicted_idx]


def preprocess_crack():
    pass

def preprocess_thermal():
    pass


trans = transforms.Compose([
    transforms.ToTensor(),
])      

import json

@app.route('/predict', methods=['POST'])
def predict():
    print(request.method)
    print('request is called')
    print(request.files['op'])
#    print(request.files['op'])
    if request.method == 'POST':
        op = request.files['op'].read().decode()
        #data = request.json
        #op = data['op']
        print('[DEBUG]op :',op)
        
        if op == 'exam':
            ret = {'crack':len(crackList),'thermal':len(rgbFiles)}
            return jsonify(ret)
        
        elif op == 'crack':
            print('[DEBUG] crack detection')
            num = int(request.files['num'].read())
            print(num,type(num))
            if num not in range(len(crackList)): 
                s = 'num:%d must be in range of %d to %d'%(num,0,len(crackList))
                return jsonify({s:0})
            filename = crackList[num]
            print('[DEBUG]filename:',filename)
            with tf.device('cpu:0'):  
                with session.as_default():
                    with session.graph.as_default():
                        vgg_pred_PIL,vgg_original_img,vgg_msk = getCrackPred(vggUNet,filename)
                        res_pred_PIL,res_original_img,res_msk = getCrackPred(resUNet,filename)
                        
                        ret = {'original':np.array(vgg_original_img).tolist(),'vgg_pred':np.array(vgg_pred_PIL).tolist(),'res_pred_PIL':np.array(res_pred_PIL).tolist()}
                        
            return jsonify(ret)
        
        elif op == 'thermal': 
            print('[DEBUG] suspected region detection')
            num = int(request.files['num'].read())
            print('[DEBUG]num:',num)
            if num not in range(len(rgbFiles)): 
                s = 'num:%d must be in range of %d to %d'%(num,0,len(rgbFiles))
                return jsonify({s:0})
            filename = rgbFiles[num]
            print('[DEBUG]filename:',filename)
            pred_PIL,r,t,l = getThermalPred(IntertUNet,filename)
            ret = {'original_RGB':np.array(r).tolist(),'original_THR':np.array(t).tolist(),'pred':np.array(pred_PIL).tolist()}
            return jsonify(ret)
        
        elif op == 'crack_pred':
            pixel_list = request.files['crack_image']
            print('[DEBUG] pixel_list type:',type(pixel_list))
            pixel_list = json.load(pixel_list)
            print('[DEBUG] pixel_list type:',type(pixel_list))
            if not isinstance(pixel_list,list):
                return jsonify({'send correct pixel list':0})
            
            crack_image = Image.fromarray(np.array(pixel_list).astype(np.uint8))
            crack_image = crack_image.resize((224,224))
            crack_image = (np.array(crack_image).astype(np.float32)/255.0).reshape((1,224,224,3))
            
            with tf.device('cpu:0'):
                with session.as_default():
                    with session.graph.as_default():
                        vggOutput = utils.predictTF(vggUNet,crack_image)
                        vggOutput = utils.TF2PIL(vggOutput)
                        resOutput = utils.predictTF(resUNet,crack_image)
                        resOutput = utils.TF2PIL(resOutput)
                        
            ret = {
                'vgg_pred':np.array(vggOutput).tolist(),
                'res_pred':np.array(resOutput).tolist()
            }
            return jsonify(ret)
        
        elif op == 'thermal_pred':
            thermal_list = request.files['thermal_image']
            rgb_list = request.files['rgb_image']
            print('[DEBUG] type:',type(thermal_list),type(rgb_list))
            
            thermal_list = json.load(thermal_list)
            rgb_list = json.load(rgb_list)
            print('[DEBUG] type:',type(thermal_list),type(rgb_list))
            
            if not isinstance(thermal_list,list) or not isinstance(rgb_list,list):
                return jsonify({'send correct pixel list':0})
            
            #rgb = Image.fromarray(np.array(rgb_list).astype(np.uint8)).resize((512,640))
            #thr = Image.fromarray(np.array(thermal_list).astype(np.uint8)).resize((512,640))
            rgb = Image.fromarray(np.array(rgb_list).astype(np.uint8)).resize((640,512))
            thr = Image.fromarray(np.array(thermal_list).astype(np.uint8)).resize((640,512))
            rgb,thr = trans(rgb).unsqueeze(0),trans(thr).unsqueeze(0)
            
            pred = utils.predictTorch(IntertUNet,rgb,thr)
            pred_PIL = utils.TorchToPIL(pred)
            ret = {
                'pred':np.array(pred_PIL).tolist()
            }
            return jsonify(ret)
        
        else:
            return jsonify({'check op code':0})

        
'''
op -> 'examples count', 'crack', 'thermal'

{'op':'examples count'} --> {'crack':len(crackList),'thermal':len(rgbFiles)}
{'op':'crack','num':123} --> {'original':vgg_original_img,'vgg_pred':vgg_pred_PIL,'res_pred_PIL':res_pred_PIL}
{'op':'thermal','num':123} --> {'original_RGB':r,'original_THR':t,'pred':pred_PIL}
'''

if __name__ == '__main__':
    app.run(host='0.0.0.0')

