# `Which Bollywood Celebrity You Look Like`

### Problem Statement :
The objective of the "Which Bollywood Celebrity You Look Like" project is to develop a deep learning-based system that can analyze an image of a user and determine which Bollywood celebrity they look like the most. The project uses Convolutional Neural Networks (CNN) and a pre-trained VGGFace model to extract features from images and identify facial characteristics of Bollywood celebrities. The project is implemented in Python programming language using TensorFlow, NumPy, Pandas, and Matplotlib. The final output is a web application using Streamlit, which allows users to upload or enter the URL of their image and get the results instantly.

### Detailed Description :

The "Which Bollywood Celebrity You Look Like" project is based on deep learning techniques and aims to provide an accurate and fun way for users to find their celebrity look-alikes. The project uses a dataset of 8600 images of Bollywood celebrities gathered from Kaggle, which includes columns such as name of celebrity and image link. The dataset is used to train a pre-trained VGGFace CNN model, which is known for its high accuracy in face recognition tasks.

### Data collection : 
The project follows a step-by-step approach to building the system. The first step is to collect the dataset from kaggle and convert all the 8600 images into publicly shareable Google Drive links and store them in a CSV file using google's app script 

### Features extraction of images : 
The next step is to extract features from all the images in the dataset using the VGGFace model and save them as a pickle file. The features extracted from the images represent the facial characteristics of the celebrities, which will be used for comparison with the user's image.

### Model implementation : 
Once the features of the dataset images are extracted, the next step is to extract the features of the user's image. The user can upload their image or enter the URL of their image using the Streamlit web application. The model extracts the features of the user's image using the same VGGFace model and cosine similarity metric. The cosine similarity metric is used to measure the similarity between the features of the user's image and the features of the celebrity images in the dataset. The image in the dataset with the highest cosine similarity value is identified as the celebrity that the user looks like.

### Streamlit web application : 
Finally, the project is implemented as a Streamlit web application. The user can simply upload their image or enter the URL of their image and get instant results. The web application uses the model to perform the facial recognition task and shows the user the name and image of the Bollywood celebrity they resemble the most. The application is user-friendly and provides an enjoyable experience for the user.

### conclusion
In conclusion, the "Which Bollywood Celebrity You Look Like" project is a fun and interactive way to explore deep learning techniques and facial recognition tasks. The project demonstrates the use of pre-trained CNN models, feature extraction techniques, and cosine similarity metric to accurately identify the celebrity look-alikes of the users. The final web application is user-friendly and provides a great user experience. The project has the potential to be extended to other domains and languages by using appropriate datasets and models.




