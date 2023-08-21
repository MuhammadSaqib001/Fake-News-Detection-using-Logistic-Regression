# Fake News Detection using Logistic Regression [Data Mining BS Project]

## Brief Description
This repository contains the code and resources for our Fake News Detection project. Our goal is to build a machine learning model to classify news articles as real or fake using Python and the `scikit-learn` library.

## Project Overview
1. **Dataset for Fake News**
   - We use a training dataset with attributes: Id, Title, Author, Text, and Label (1: Fake, 0: Real).
   - Source of Dataset: Fake-News.csv

2. **Dataset Analysis**
   - The dataset shape is (3000, 6) with 1521 fake and 1479 real news articles.

3. **Data Preprocessing**
   - Cleaning missing values.
   - Replacing numbers and punctuations with whitespaces.
   - Removing stopwords and reducing words to their root forms.
   - Converting textual data into numerical vectors.
   - Selecting desired features.

4. **Train & Test Split**
   - Preprocessed data is split into training and testing sets (80% training, 20% testing).

5. **Logistic Regression**
   - A Logistic Regression model is trained to classify news articles.
   - Training accuracy: 94%, Test accuracy: 90%.

6. **Evaluation**
   - Confusion matrix is used to assess model accuracy due to dataset imbalance.

7. **Prediction of Trained Model**
   - Model is validated using sample input strings to evaluate its performance.

![data](https://github.com/MuhammadSaqib001/Fake-News-Detection-using-Logistic-Regression/blob/main/images/1%20(1).png)

![data](https://github.com/MuhammadSaqib001/Fake-News-Detection-using-Logistic-Regression/blob/main/images/1%20(2).png)

![data](https://github.com/MuhammadSaqib001/Fake-News-Detection-using-Logistic-Regression/blob/main/images/1%20(3).png)

![data](https://github.com/MuhammadSaqib001/Fake-News-Detection-using-Logistic-Regression/blob/main/images/1%20(4).png)
