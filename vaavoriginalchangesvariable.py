# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 05:05:28 2022

@author: HARSHITH REDDY
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 04:21:48 2022

@author: HARSHITH REDDY
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

df=pd.read_excel(r"C:/Users/HARSHITH REDDY/Downloads/assignment_data.xlsx")

df.shape

df.info

df.describe

df.columns



### Identify duplicates records in the data ###
duplicate = df.duplicated()
duplicate
sum(duplicate)  # no duplicates in our data
# Removing Duplicates
data = df.drop_duplicates()


# check for count of null in each column
df.isna().sum()

# Create an imputer object that fills 'Nan' values
# Mean and Median imputer are used for numeric data 
# Mode is used for discrete data 

# for Mean, Meadian, Mode imputation we can use Simple Imputer or data.fillna()
from sklearn.impute import SimpleImputer
# Mode Imputer for column description since it is categorical data
mode_imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
df["description"] = pd.DataFrame(mode_imputer.fit_transform(df[["description"]]))

# check for count of null in each column
df.isna().sum() #description column got imputed now no cells are empty



#description column data is collecting in a list
description_list=df.description

#description_list=list(description_list)

############################cleaning job description by removing all html tags##################
import re

regex = re.compile(r'<[^>]+>')
n=0
b=len(description_list)
for r in range(b):
    def remove_html(string):
        return regex.sub('', string)
    
    text=description_list[n]
    description_list[n]=remove_html(text)
    n=n+1
########################################################################


#description column data is collecting in a list
#description_list=df.description

description_list=list(description_list)


#df['description_new']=description_list
q=str(description_list)


# writng reviews in a text file 
# saving the reviews in a textt file i.e., oneplus as text file
with open("readme.txt", "w", encoding='utf8') as output:
    output.write(q)


import os
os.getcwd()

	

# Joinining all the reviews into single paragraph 
ip_rev_string = q#" ".join(str(description_list))
import re
import nltk
# from nltk.corpus import stopwords

# Removing unwanted symbols incase if exists
ip_rev_string = re.sub("[^A-Za-z" "]+", " ", ip_rev_string).lower()
# ip_rev_string = re.sub("[0-9" "]+"," ", ip_rev_string)

# words that contained in the description attribute
job_description_words = ip_rev_string.split(" ")

#ip_reviews_words = ip_reviews_words[1:]
#print(ip_reviews_words)
#f=len(ip_reviews_words)
#str_ip_reviews_words=str(ip_reviews_words)








#TFIDF
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(job_description_words, use_idf=True, ngram_range=(1, 1))
X = vectorizer.fit_transform(job_description_words)

with open("C:/Users/HARSHITH REDDY/Desktop/stopwords_en.txt", "r") as sw:
    stop_words = sw.read()
    
stop_words = stop_words.split("\n")

stop_words.extend(["xa"])

job_description_words = [w for w in job_description_words if not w in stop_words]

# Joinining all the reviews into single paragraph 
ip_rev_string = " ".join(job_description_words)

# WordCloud can be performed on the string inputs.
# Corpus level word cloud
from wordcloud import WordCloud, STOPWORDS
wordcloud_ip = WordCloud(background_color='White',
                      width=1800,
                      height=1400
                     ).generate(ip_rev_string)
plt.imshow(wordcloud_ip)
#### There is no need of doing positive and negative words word cloud because job description always have neutral words##################################
###########################positive words word cloud#################
## positive words # Choose the path for +ve words stored in system
#with open("C:/Users/HARSHITH REDDY/Desktop/stopwords_en.txt","r") as pos:
#  poswords = pos.read().split("\n")

## Positive word cloud
## Choosing the only words which are present in positive words
#ip_pos_in_pos = " ".join ([w for w in ip_reviews_words if w in poswords])
#from wordcloud import WordCloud, STOPWORDS
#wordcloud_pos_in_pos = WordCloud(
#                      background_color='White',
#                      width=1800,
#                      height=1400
#                     ).generate(ip_pos_in_pos)
#plt.figure(2)
#plt.imshow(wordcloud_pos_in_pos)

## negative words Choose path for -ve words stored in system
#with open("C:\\Users\\kaval\\OneDrive\\Desktop\\360digit\\datatypes\\negative-words.txt", "r") as neg:
#  negwords = neg.read().split("\n")

## negative word cloud
## Choosing the only words which are present in negwords
#ip_neg_in_neg = " ".join ([w for w in ip_reviews_words if w in negwords])
#from wordcloud import WordCloud, STOPWORDS
#wordcloud_neg_in_neg = WordCloud(
#                      background_color='black',
#                      width=1800,
#                      height=1400
#                     ).generate(ip_neg_in_neg)
#plt.figure(3)
#plt.imshow(wordcloud_neg_in_neg)


# wordcloud with bigram
nltk.download('punkt')
from wordcloud import WordCloud, STOPWORDS

WNL = nltk.WordNetLemmatizer()

# Lowercase and tokenize
text = ip_rev_string.lower()

# Remove single quote early since it causes problems with the tokenizer.
text = text.replace("'", "")

tokens = nltk.word_tokenize(text)
text1 = nltk.Text(tokens)

# Remove extra chars and remove stop words.
text_content = [''.join(re.split("[ .,;:!?‘’``''@#$%^_&*()<>{}~\n\t\\\-]", word)) for word in text1]

# Create a set of stopwords
stopwords_wc = set(STOPWORDS)
customised_words = ['Coimbatore'] # If you want to remove any particular word form text which does not contribute much in meaning

new_stopwords = stopwords_wc.union(customised_words)

# Remove stop words
text_content = [word for word in text_content if word not in new_stopwords]

# Take only non-empty entries
text_content = [s for s in text_content if len(s) != 0]

# Best to get the lemmas of each word to reduce the number of similar words
text_content = [WNL.lemmatize(t) for t in text_content]

# nltk_tokens = nltk.word_tokenize(text)  
bigrams_list = list(nltk.bigrams(text_content))
print(bigrams_list)

dictionary2 = [' '.join(tup) for tup in bigrams_list]
print (dictionary2)

# Using count vectoriser to view the frequency of bigrams
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(ngram_range=(2, 2))
bag_of_words = vectorizer.fit_transform(dictionary2)
vectorizer.vocabulary_

sum_words = bag_of_words.sum(axis=0)
words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)
l= words_freq[:100]
print(l)

# Generating wordcloud
words_dict = dict(words_freq)
WC_height = 1000
WC_width = 1500
WC_max_words = 100
wordCloud = WordCloud(max_words=WC_max_words, height=WC_height, width=WC_width, stopwords=new_stopwords)

wordCloud.generate_from_frequencies(words_dict)
plt.figure(4)
plt.title('Most frequently occurring bigrams connected by same colour and font size')
plt.imshow(wordCloud, interpolation='bilinear')
plt.axis("off")
plt.show()

###############clustering######################


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

documents = df['description'].values.astype("U")

vectorizer = TfidfVectorizer(stop_words='english')
features = vectorizer.fit_transform(documents)
df.info
k = 34 #decided k value by thumb rule
model = KMeans(n_clusters=k, init='k-means++', max_iter=100, n_init=1)
model.fit(features)

df['cluster'] = model.labels_

df.head()

# output the result to a text file.

clusters = df.groupby('cluster')
clustered_data= 'clustered_data.xlsx'  
df.to_excel(clustered_data)


print("Cluster centroids: \n")
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()

for i in range(k):
    print("Cluster %d:" % i)
    for j in order_centroids[i, :10]: #print out 10 feature terms of each cluster
        print (' %s' % terms[j])
    print('------------')