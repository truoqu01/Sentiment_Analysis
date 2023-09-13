import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from underthesea import word_tokenize, pos_tag, sent_tokenize
#import regex
import re
import pickle
import string
from my_funcion import *
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
st.set_option('deprecation.showPyplotGlobalUse', False)


data = pd.read_csv('Sendo_reviews.csv')
data_sub = pd.read_csv('data_sub.csv', encoding='utf8')


uploaded_file = st.file_uploader("Choose a file", type=['csv'])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file, encoding='utf8')
    data.to_csv("data.csv", index = False)


data_sub.dropna(inplace=True)
##LOAD EMOJICON
file = open('emojicon.txt', 'r', encoding="utf8")
emoji_lst = file.read().split('\n')
emoji_dict = {}
for line in emoji_lst:
    key, value = line.split('\t')
    emoji_dict[key] = str(value)
file.close()
#################
#LOAD TEENCODE
file = open('teencode.txt', 'r', encoding="utf8")
teen_lst = file.read().split('\n')
teen_dict = {}
for line in teen_lst:
    key, value = line.split('\t')
    teen_dict[key] = str(value)
file.close()
###############
#LOAD TRANSLATE ENGLISH -> VNMESE
file = open('english-vnmese.txt', 'r', encoding="utf8")
english_lst = file.read().split('\n')
english_dict = {}
for line in english_lst:
    key, value = line.split('\t')
    english_dict[key] = str(value)
file.close()
################
#LOAD wrong words
file = open('wrong-word.txt', 'r', encoding="utf8")
wrong_lst = file.read().split('\n')
file.close()
#################
#LOAD STOPWORDS
file = open('vietnamese-stopwords.txt', 'r', encoding="utf8")
stopwords_lst = file.read().split('\n')
file.close()


data_sub = pd.read_csv('data_sub.csv')
data_sub.dropna(inplace=True)


#Build Model
source = data_sub['processed_content_wt']
target = data_sub['class']

text_data = np.array(source)

count = CountVectorizer(max_features=6000)
count.fit(text_data)
bag_of_words = count.transform(text_data)

X = bag_of_words.toarray()

y = np.array(target)


X_train, X_test, y_train, y_test = train_test_split(X ,y , test_size=0.3, random_state=42)

from imblearn.over_sampling import SMOTE
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)



clf = MultinomialNB()
model = clf.fit(X_resampled, y_resampled)
y_pred = clf.predict(X_test)


score_train = model.score(X_train, y_train)
score_test = model.score(X_test,y_test)

acc = accuracy_score(y_test,y_pred)
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
cr = classification_report(y_test, y_pred)

#5. Save models
# luu model classication
pkl_filename = "sentiment_analysis_model.pkl"  
with open(pkl_filename, 'wb') as file:  
    pickle.dump(model, file)
  
# luu model CountVectorizer (count)
pkl_count = "count_model.pkl"  
with open(pkl_count, 'wb') as file:  
    pickle.dump(count, file)


#6. Load models 
# Đọc model
# import pickle
with open(pkl_filename, 'rb') as file:  
    sentiment_analysis_model = pickle.load(file)
# doc model count len
with open(pkl_count, 'rb') as file:  
    count_model = pickle.load(file)


menu = ["Business Objective", "Build Project", "New Prediction"]
choice = st.sidebar.selectbox('Menu', menu)
if choice == 'Business Objective':    
    st.subheader("Business Objective")
    st.write("""
    ###### Based on the history of previous customer reviews => Data collected from the customer comments and reviews section at https://www.sendo.vn/…
    """)  
    st.write("""###### => Goal/problem: Building a prediction model helps salespeople know quickly about customer feedback about their products or services (positive, negative or neutral). This helps sellers know the business situation, understand customer opinions, thereby helping them improve their services and products..""")
    st.image("Sentiment_Analysis.png")
    st.image("Sendo.jpg")

elif choice == 'Build Project':
    st.subheader("Build Project")
    st.write("##### 1. Some data")
    st.dataframe(data_sub.head(3))
    st.dataframe(data_sub.tail(3))  
    st.write('0 is negative, 1 is normal, 2 is positive')

    st.write("##### 2. Visualize")
    st.title("Biểu đồ Word Cloud")
    st.image("word_cloud.png")
    st.title("Biểu đồ Histogram")
    # Vẽ biểu đồ histogram bằng Seaborn
    sns.histplot(data_sub['rating'], bins=10, kde=True)
    plt.xlabel('Rating')
    plt.ylabel('Tần suất')
    plt.title('Phân phối các đánh giá')
    
    # Hiển thị biểu đồ lên Streamlit
    st.pyplot()

    st.write("##### 3. Build model...")
    st.write("##### 4. Evaluation")

    st.code("Score train:"+ str(round(score_train,2)) + " vs Score test:" + str(round(score_test,2)))
    st.code("Accuracy:"+str(round(acc,2)))

    st.write("###### Confusion matrix:")
    st.code(cm)

    st.write("###### Classification report:")
    st.code(cr)


elif choice == 'New Prediction':
    st.subheader("Select data")
    flag = False
    lines = None
    type = st.radio("Upload data or Input data?", options=("Upload", "Input"))

    if type=="Upload":
        # Upload file
        uploaded_file_1 = st.file_uploader("Choose a file", type=['txt', 'csv'])
        if uploaded_file_1 is not None:
            lines = pd.read_csv(uploaded_file_1, header=None)
            st.dataframe(lines)
            # st.write(lines.columns)
            lines = lines[0]     
            flag = True 

    if type=="Input":        
        email = st.text_area(label="Input your content:")
        if email!="":
            lines = np.array([email])
            flag = True
    
    if flag:
        st.write("Content:")
        if len(lines)>0:
            st.code(lines)
            processed_lines = [preprocess_text_1(line, emoji_dict, teen_dict, wrong_lst, english_dict, stopwords_lst) for line in lines]
            tokenized_lines = [word_tokenize(line) for line in processed_lines]
            x_new = count_model.transform(processed_lines)        
            y_pred_new = sentiment_analysis_model.predict(x_new)       
            st.code("New predictions (0: negative, 1: normal, 2: positive): " + str(y_pred_new))




