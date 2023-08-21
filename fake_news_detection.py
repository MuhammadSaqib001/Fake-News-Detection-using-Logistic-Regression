#!/usr/bin/env python
# coding: utf-8

# Data Mining Project "FAKE News Detection"
"""Project :- Group Members

   --> 19I-0475 Haseeb Ramzan CS_F 
   --> 19I-0504 Amjid Arshad CS_F
   --> 19I-0494 Muhammad Saqib CS_F """
# ### Project Description

# In order to accurately classified collection of news as real or fake we have to build a machine learning model.

# To deals with the detection of fake or real news, we will develop the project in python with the help of ‘sklearn’, we will use ‘TfidfVectorizer’ in our news data which we will gather from online media.

# After the first step is done, we will initialize the classifier, transform and fit the model. In the end, we will calculate the performance of the model using the appropriate performance matrix/matrices. Once will calculate the performance matrices we will be able to see how well our model performs.

# Dataset used - https://www.kaggle.com/c/fake-news/data/train.csv

# Dataset Description
# train.csv: A full training dataset with the following attributes :-

#  1. id: unique id for a news article
#  2. title: the title of a news article
#  3. author: author of the news article
#  4. text: the text of the article; could be incomplete
  
#     label: a label that marks the article as potentially unreliable
#     1: FAKE
#     0: REAL

# In[314]:


from IPython.display import Image
Image(filename="myproject.jpg")


# These are the libraries which are to be loaded for correct running of the program . Anaconda integrated with NLTK is necessary . Otherwise it would end up giving errors

# In[315]:


#Importing Libraries to be used 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

import re #Regular expressions 
import nltk #Natural Language Toolkit
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn import metrics
import itertools


# ### 1. DATA Loading & Analysis

# In this step , data is being loaded from a csv file and being analyzed for the preprocessing step .

# In[316]:


#Data Set loaded as Pandas Dataset
data = pd.read_csv("dataset.csv",nrows=3000)
data.head()


# In[317]:


data['text corpus'] = data['title']+' '+data['author']+' '+data['text']
data = data[["id", "title","author","text","text corpus", "label"]]
data.head()


# In[391]:


# Data and Output Analysis
print(data['label'].value_counts())
data.shape
graph_sc = {'id': [1,2,3,4,5,6,7,8,9,10],
        'label': [0,1,1,1,1,1,0,0,1,0]
       }
  


# ### 2. DATA PreProcessing Step 
Part2(1). Data Cleaning
   Detecting all the rows with NULL values and then Ignoring all those rows 
# In[319]:


data.isnull().sum()


# In[320]:


data = data[data['text corpus'].notna()]

data.isnull().sum()


# In[398]:


df = pd.DataFrame(graph_sc,columns=['id','label'])
df.plot(x ='id', y='label',kind = 'scatter')
plt.xlabel('ID of the text')
plt.ylabel('FAKE or REAL')
plt.title('FAKE-News Dataset')
plt.show()

Part2(2). 
   1. Remove Numbers and Punctutions with whitespaces
   2. Remove all the words which have been detected as the stopwords
   3. Reduce the word to it's origin or root word
# In[325]:


def make_inorder(ser, match_name, default=None, regex=False, case=False):
    """ Search a series for text matches.
    Based on code from https://www.metasnake.com/blog/pydata-assign.html
    """
    seen = None
    for match, name in match_name:
        mask = ser.str.contains(match, case=case, regex=regex)
        if seen is None:
            seen = mask
        else:
            seen |= mask
        ser = ser.where(~mask, name)
    if default:
        ser = ser.where(seen, default)
    else:
        ser = ser.where(seen, ser.values)
    return ser


# In[18]:


port_stem = PorterStemmer()

def data_cleaner(text_values):
    Ntext_values = re.sub('[^a-zA-Z]',' ',text_values) 
    Ntext_values = Ntext_values.lower()
    Ntext_values = Ntext_values.split()
    Ntext_values = [port_stem.stem(word) for word in Ntext_values if not word in stopwords.words('english')]
    Ntext_values = ' '.join(Ntext_values)
    return Ntext_values


# In[ ]:


data['text corpus'] = data['text_corpus'].apply(cleaner)


# In[322]:


X = data['text corpus'].values
X


# In[323]:


Y = data['label'].values
Y

Part2(3). Converting Textual data to Numerical data
# In[324]:


vectorizer = TfidfVectorizer()
vectorizer.fit(X)

X = vectorizer.transform(X)
print(X)

Part2(4). Feature Selection
    Ignoring all the features which are not required . Text_Corpus had all the data we need i.e., concatenated title , author and text . We have some extra features like title , author and text .
# In[328]:


data = data[["id","text corpus", "label"]]
data.head(10)


# ### 3. Splitting :- TRAIN & TEST Data
Data from csv file is being spiltted into two parts with 0.18 as Training data and 0.82 as Test data . Also , this split is random . Each time , TEST and TRAIN data changes .
# In[335]:


def train_test_split(data_train,data_test,folder_train,foler_test) :
    
    os.mkdir(folder_train)
    train_ind=list(data_train.index)
    
    # Train folder
    for i in tqdm(range(len(train_ind))):
        os.system('cp '+data_train[train_ind[i]]+' ./'+ folder_train + '/'  +data_train[train_ind[i]].split('/')[2])
   
    # Test folder
    for j in tqdm(range(len(test_ind))):
        os.system('cp '+data_test[test_ind[j]]+' ./'+ folder_test + '/'  +data_test[test_ind[j]].split('/')[2])


# In[282]:


#Splitting DataSet in Train && Test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.18, stratify=Y, random_state=124)


# ### 4. Machine Learning Model :- LOGISTIC Regression
Logistic Regression is being used in order to classify the data vector as FAKE or REAL .
# In[326]:


def sigmoid(x):
    # Activation function used to map any real value between 0 and 1
    return 1 / (1 + np.exp(-x))

def net_input(theta, x):
    # Computes the weighted sum of inputs
    return np.dot(x, theta)

def probability(theta, x):
    # Returns the probability after passing through sigmoid
    return sigmoid(net_input(theta, x))


# In[327]:


def cost_function(self, theta, x, y):
    # Computes the cost function for all the training samples
    m = x.shape[0]
    total_cost = -(1 / m) * np.sum(
        y * np.log(probability(theta, x)) + (1 - y) * np.log(
            1 - probability(theta, x)))
    return total_cost

def gradient(self, theta, x, y):
    # Computes the gradient of the cost function at the point theta
    m = x.shape[0]
    return (1 / m) * np.dot(x.T, sigmoid(net_input(theta,   x)) - y)


# In[283]:


model1 = LogisticRegression()

model1.fit(X_train, Y_train)


# In[284]:


# Accuracy Score on Training Data
X_train_prediction = model1.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
np.set_printoptions(threshold=np.inf)
print(X_train_prediction)


# In[285]:


# Accuracy Score on Test Data
X_test_prediction = model1.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
X_test_prediction


# In[286]:


print('Accuracy score on the training data: ',training_data_accuracy)
print('Accuracy score on the test data: ',test_data_accuracy)


# ### 4. Model Evaluation :- CONFUSION Matrix

# In[333]:


def classification_accuracy(classification_scores, true_labels):
    """ Returns the fractional classification accuracy for a batch of N predictions """
    pred_labels = []  # Will store the N predicted class-IDs
    for row in classification_scores:
        pred_labels.append(np.argmax(row))

    num_correct = 0
    for i in range(len(pred_labels)):
        if pred_labels[i] == true_labels[i]:
            num_correct += 1
    return num_correct / len(true_labels)


# In[287]:


def plot_confusion_matrix(cm, classes,normalize=False,title='Confusion Matrix',cmap=plt.cm.Blues):
  plt.imshow(cm, interpolation='nearest', cmap=cmap)
  plt.title(title)
  plt.colorbar()
  tick_marks = np.arange(len(classes))
  plt.xticks(tick_marks, classes, rotation=45)
  plt.yticks(tick_marks, classes)

  thresh = cm.max() / 2.
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
      plt.text(j, i, cm[i, j],horizontalalignment="center",color="white" if cm[i, j] > thresh else "black")
  plt.tight_layout()


# In[336]:


cm = metrics.confusion_matrix(Y_test, X_test_prediction)
plot_confusion_matrix(cm, classes=['Fake', 'Real'])


# ### 5. Prediction using Trained MODEL 

# In[331]:


X_new = X_test[100]

prediction = model1.predict(X_new)

if (prediction[0] == 0):
  print('The news is Real')
else:
  print('The news is Fake & Unreliable')


# ###                                                                THE END
