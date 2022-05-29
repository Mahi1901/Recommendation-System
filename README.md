# Recommendation-System
What is a Recommender System? 

A recommendation system is an information filtering system that seeks to predict the rating given by a user to an item. This predicted rating is further used to recommend items to the user. 
The item for which the predicted rating is high will be recommended to the user. It is used by almost all of the major companies to enhance their business and to enrich user experience 
like YouTube for recommending videos, Amazon & Ebay for recommending products, Netflix for recommending Movies, Airbnb for recommending rooms and hotels, etc. 

Among the various filtering types for recommendation system,  content-based filtering has been used for the recommendation system model. 

How content-based Filtering works? 

Content based filtering is similar in approach with classical machine learning techniques. It needs a way to represent an item Ij and a user Ui. 
It needs to collect information about an item Ij and a user Ui and further create features of both user Ui and item  Ij. 
These features are then combined which are then input to a Machine Learning model for training. Here, label will be Aij, which is the corresponding rating given by a user Ui on item Ij. 

Dataset: 

I've worked on the Netflix dataset and built a model to recommend movies to the end users. 

Dataset Description: Netflix data set. 

This data set consists of: 

5543 records of Movie/TV Shows. 

In total 16 attributes describing each Movie/TV Show. 

Acknowledgements: 

The dataset was downloaded from Kaggle along with other datasets, together merged to make a new dataset. 

Tech Stack: 
HTML 
CSS 
Bootstrap 
JavaScript 
Python 
Django 
SQLite3 

Prerequisites: 

Installations: 

Following libraries were used in the app: 

 

Follow the steps to install following libraries: 

pip install django 

pip install pandas 

pip install numpy 

pip install scikit-learn 

pip install SpeechRecognition 

pip install imdbpy 

Preprocessing: 

The recommendation model uses Count Vectorizer, Cosine Similarity, Tim Sort and Linear Search algorithms. 

COUNT VECTORIZER: 

Count Vectorizer is used to convert a collection of text documents to a vector of term/token counts. The scikit-learn library in Python provides the Count Vectorizer tool. It transforms the given text into a vector on the basis of the frequency of each word in the entire document, hence, creating a matrix in which each unique word is represented by a column of the matrix, and each text sample from the document is a row in the matrix. The value of each cell is the count of the word in that particular text sample.  Inside Count Vectorizer, these words are not stored as strings. Rather, they are given a particular index value. 

COSINE SIMILARITY: 

The cosine similarity concept that has been used, quantifies the similarity between two or more vectors. It is the cosine of the angle between vectors. Cosine similarity is  commonly used as similarity measurement technique that can be found in widely used libraries and tools such as Matlab, SciKit-Learn, TensorFlow etc. Cosine Similarity is a value that is bound by a constrained range of 0 and 1. The vector representations of the documents can then be used within the cosine similarity formula to obtain a quantification of similarity.  

 

TIM SORT: 

In this recommendation model, Tim sort is used to sort the Movie/TV Shows based on the enumerated cosine similarities between both the vector quantities. 

Python sorted () function was used that  utilizes the Tim sort algorithm which is a hybrid sorting algorithm, derived from merge sort and insertion sort. Sorted () method sorts the given sequence as well as set and dictionary (which is not a sequence) either in ascending order or in descending order (does Unicode comparison for string char by char) and always return a sorted list.  

 

SEQUENTIAL SEARCH: 

In this recommendation model, sequential search algorithm has been used,  where it start searching from one end and have checked every element in the list until the desired element is found. 

 

Output: 

The output of this recommendation system is to display the top recommended Movies/TV Shows on the basis of the Movies/TV Show searched by the user using the following algorithm: 

 

 

Future scope: 

The future scope of this project is to include the following: 

User Wishlist 

Kids section 

Feedback form 

Selection of website language 

 

 

 

 

 
