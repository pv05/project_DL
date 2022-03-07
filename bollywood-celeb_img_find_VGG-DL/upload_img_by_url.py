import streamlit as st 
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
import numpy as np 
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import cv2
import os
from mtcnn import MTCNN
from PIL import Image
import pandas as pd 
import requests, io
import matplotlib.pyplot as plt 


feature_list = np.array(pickle.load(open('embedding_actors.pkl','rb')))
filenames = pickle.load(open('actors_filenames.pkl','rb'))

model = VGGFace(model='resnet50',include_top=False,input_shape=(224,224,3),pooling='avg')
df = pd.read_csv('final_imglink_bollywood.csv')

st.markdown('<h1>Check Which Bollywood Celebrity You Look Like</h1>',unsafe_allow_html=True)

def extractfeature():
    image = Image.fromarray(face)
    image = image.resize((224,224))
    face_array = np.asarray(image)
    face_array = face_array.astype('float32')
    expanded_img = np.expand_dims(face_array,axis=0)
    preprocess_img = preprocess_input(expanded_img)
    results = model.predict(preprocess_img).flatten()

    return results

detector = MTCNN()

def linkchecker(link):
    pasteboard  = link.split('/')[2]
    if pasteboard == 'pasteboard.co':
        link = link.replace('pasteboard.co','gcdnb.pbrd.co/images')
        return link
    else:
        return link


st.markdown('<h6>Please upload .jpg image format</h6>',unsafe_allow_html=True)
st.write("If you don't have image url then upload your image [Here](https://pasteboard.co) and paste your link below")
upload_img = st.text_input("Your link must be look like - https://pasteboard.co/6kBGWcSkM3ep.jpg ")

try:
    if st.button('Check'):
        upload_img = linkchecker(upload_img)
        response = requests.get(upload_img).content
        sample_img = plt.imread(io.BytesIO(response), format='JPG')
        result = detector.detect_faces(sample_img)
        if len(result) != 0:
            x,y,width,height = result[0]['box']
            face = sample_img[y:y+height,x:x+width]
            results = extractfeature()
            similarity = [] # it store our give image's similarty score
            for i in range(len(feature_list)):
                similarity.append(cosine_similarity(results.reshape(1,-1),feature_list[i].reshape(1,-1))[0][0])
            index_pos = sorted(list(enumerate(similarity)),reverse=True,key=lambda x:x[1])[0][0]
            imglink = filenames[index_pos].split('/')[-1] 
            final_img  = df[df['name'] == imglink]['img_link'].values[0]
            img_name = df[df['name'] == imglink]['name'].values[0].split('.')[0].replace('_',' ')

            
            st.image(final_img)
            st.markdown(f'<h6>You look like {img_name}</h6>',unsafe_allow_html=True)
        
        else:
            st.error('Opps Sorry!! Choose those image which has clearly show face')

except:
    st.error('Check your image url again ')
    st.write('your url must be look like : https://pasteboard.co/6kBGWcSkM3ep.jpg ')

