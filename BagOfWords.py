import sqlite3
import pandas as pd
import numpy as np
import nltk
import string
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import seaborn as sns
from nltk.stem.porter import PorterStemmer

#Using sqlite table to read data
con = sqlite3.connect('C:\\Users\\Maithreyi.Prabhu\\Downloads\\amazon-fine-food-reviews\\database.sqlite')
filtered_data = pd.read_sql_query("""
SELECT * FROM
Reviews
WHERE SCORE != 3
""",con)
print(filtered_data)

#Rows where reviews have score > 3, give  them a positive rating
#Rows where reviews have score < 3, give them a negative rating

def partition(x):
    if x<3:
        return 'Negative'
    return 'Positive'

#Change the rows with score less than 3 as negative and greater than 3 as positive
actualScore = filtered_data['Score']
print("Printing actual score variable here",actualScore)
PositiveNegative = actualScore.map(partition)
print(PositiveNegative)
filtered_data['Score'] = PositiveNegative
print("Printing shape and first 5 columns of the filtered data got")
print(filtered_data.shape)
print(filtered_data.head())

#Sorting data according to product ID in ascending order and remove the duplicates where userID and profilename and time togehter is exactl the same

sorted_data = filtered_data.sort_values('ProductId',axis=0, ascending=True)
final_data = sorted_data.drop_duplicates(subset={'UserId', "ProfileName","Time","Text"}, keep='first', inplace=False)
print("Total percentage of data present after removing the duplicates is:", end=" ")
print(((final_data['Id'].size*1.0)/(filtered_data['Id'].size))*100)

# The other Helpfulness Numerator < = Helpfulness Denominator
final = final_data[final_data['HelpfulnessNumerator'] <= final_data['HelpfulnessDenominator']]
print("Final after removing the rows which has helpfulnessNumerator <= Helpfulness Denominator")
print(final)
print("Shape : ",final.shape)
print("Total percentage of data present after removing hte second data cleaning :", end=" ")
print(((final['Id'].size*1.0)/(filtered_data['Id'].size))*100)

count_vect = CountVectorizer() #Initialize the count vector
final_counts = count_vect.fit_transform(final['Text'].values)
print("Final count vdetails are")
print(final_counts)
print(type(final_counts))
