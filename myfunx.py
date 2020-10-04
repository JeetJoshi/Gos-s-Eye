import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing import image
import cv2


my_model1 = keras.models.load_model('my_im.h5')





def chop(tada):
    x = tada
    x1 = []
    t = 0
    for i in range(1,11):
        for j in range(1,11):
            x1.append(x[(i-1)*int(x.shape[0]/10):int(x.shape[0]/10)*i,(j-1)*int(x.shape[1]/10):j*int(x.shape[1]/10),0:3])
            cv2.imwrite(f'./data/{t+1000}.jpg',x[(i-1)*int(x.shape[0]/10):int(x.shape[0]/10)*i,(j-1)*int(x.shape[1]/10):j*int(x.shape[1]/10),0:3])
            t += 1





def compare(H1,H2):
    Op = []
    for i in range(len(H1)):
        for j in range(len(H2)):
            if (H1[i][0]) == (H2[j][0]):
                Op.append(H1[i][0])
    return Op






def generate(x,x1,Op):
    for t in range(len(Op)):        
        temp = Op[t] + 1
        if temp%10==0:
            p2 = temp%10 + 10
        
        else:
            p1 = temp//10 +1
            p2 = temp%10
        print(p1,p2)
        for i in range(1,11):
            for j in range(1,11):
                if i == p1 and j==p2:
                    x[(i-1)*int(x.shape[0]/10):int(x.shape[0]/10)*i,(j-1)*int(x.shape[1]/10):j*int(x.shape[1]/10)] = [0,0,255]
                    x[(i-1)*int(x.shape[0]/10)+5:i*int(x.shape[0]/10)-5,(j-1)*int(x.shape[1]/10)+5:j*int(x.shape[1]/10)-5]  = x1[(i-1)*int(x1.shape[0]/10)+5:int(x1.shape[0]/10)*i-5,(j-1)*int(x1.shape[1]/10)+5:j*int(x1.shape[1]/10)-5]
    x = cv2.resize(x,(1227,519))
    return x





def predict(tada):
    chop(tada)
    H1 = []
    H2 = []
    n = 0
    for i in range(0,100):
        img_path = (f'./data/{i+1000}.jpg')
        img = image.load_img(img_path, target_size=(150, 150))
        x = image.img_to_array(img)/255
        x = np.expand_dims(x, axis=0)
        name1 = my_model1.predict(x)
        name2 = my_model1.predict(x)          
        if name1[0] > 0.01:
                pass               
        else:
            H1.append([n,img])               
        if name2[0] > 0.01:
            pass            
        else:
            H2.append([n,img])               

        n += 1
    Op = compare(H1,H2)
    print(Op)       
    return Op



def run():
    p = (generate(cv2.imread('snapshot11.jfif'),cv2.imread('snapshot11.jfif'),predict(cv2.imread('snapshot11.jfif'))))
    p = cv2.cvtColor(p, cv2.COLOR_BGR2RGB)

    return p
    

    

