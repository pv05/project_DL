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
import urllib.request


# feature_list = np.array(pickle.load(open('embedding_actors.pkl','rb')))
# filenames = pickle.load(open('actors_filenames.pkl','rb'))

feature_list = np.array(pickle.load(urllib.request.urlopen('https://drive.google.com/file/d/16Dsj2JJ41eQQ1F9MvCXfepH3pSihYa1M/view?usp=sharing')))
filenames = pickle.load(urllib.request.urlopen('https://drive.google.com/file/d/14SZ9ZFrBeQ8DNx5qaJHhLp4MSpcYwmBj/view?usp=sharing'))

model = VGGFace(model='resnet50',include_top=False,input_shape=(224,224,3),pooling='avg')
# df = pd.read_csv('final_imglink_bollywood.csv')
df = pd.read_csv('https://raw.githubusercontent.com/pv05/project_DL/main/bollywood-celeb_img_find_VGG-DL/final_imglink_bollywood.csv')

st.title('Check Which Bollywood Celebrity You Look Like')

def save_upploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploads', uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
            return 1
    except:

        return 0

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
upload_img = st.file_uploader('Choose an image',type=['jpg','png'])

if upload_img is not None:
    save_upploaded_file(upload_img)
    sample_img = cv2.imread(os.path.join('uploads',upload_img.name))
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




