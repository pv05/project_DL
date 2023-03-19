# `FASHION RECOMMANDER SYSTEM`

### Problem Statement :
Online shopping has become a popular way for people to buy clothes, accessories and other fashion items. However, with a large number of products available online, it can be difficult for users to find what they are looking for. One way to help users is to recommend products that are similar to what they are interested in. This project aims to create a Fashion Recommender System that suggests similar products to users based on an uploaded image.

### Detailed Description :
The Fashion Recommender System is a deep learning based project that uses ResNet50 and NearestNeighbors algorithm to recommend similar products to users. The project is implemented using Python programming language and popular libraries such as TensorFlow, NumPy, Pandas, and Matplotlib.

### Data collection : 
The dataset used in the project is gathered from Kaggle, which contains a collection of 44,000 images of fashion products. To extract features from these images, a pre-trained ResNet50 model is used. The ResNet50 model is a popular convolutional neural network (CNN) that is widely used for image recognition and classification tasks. The ResNet50 model has been trained on a large-scale dataset called ImageNet, which contains over a million images.

### Features extraction : 
To create the Fashion Recommender System, first, the features of all 44,000 product images are extracted using the ResNet50 model, and saved as a pickle file. Then, when a user uploads an image of a product they are interested in, the features of that image are also extracted using the ResNet50 model. 

### Algorithm apply : 
Next, the NearestNeighbors algorithm is used to find the top 5 most similar product images from the dataset based on the extracted features of the uploaded image. 

The NearestNeighbors algorithm is a popular machine learning algorithm that is used for similarity search and clustering tasks. The algorithm works by finding the k-nearest neighbors of a given query point in a dataset. In this project, the k is set to 5, which means that the algorithm finds the top 5 most similar product images to the uploaded image.

### Streamlit application created : 
Finally, a Streamlit web application is created where users can drop their product image from their local system and the model will recommend the top 5 similar product images.

The Streamlit web application is a user-friendly interface that allows users to upload an image of a product they are interested in and get recommendations for similar products. The web application is created using Streamlit, which is an open-source Python library that allows users to create interactive web applications easily.

### Conclusion : 
In conclusion, the Fashion Recommender System is a deep learning project that uses ResNet50 and NearestNeighbors algorithm to recommend similar products to users based on an uploaded image. The project is implemented using Python programming language and popular libraries such as TensorFlow, NumPy, Pandas, and Matplotlib. The project can be useful for online fashion retailers who want to provide personalized recommendations to their customers based on their preferences.




