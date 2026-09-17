import numpy as np
import pandas as pd
#We use pandas to create the data frames, because the data is in csv format and it is not easy to put that data into structured format

from sklearn.model_selection import train_test_split
#we need this, because we need to split the data into training and testing data

from sklearn.feature_extraction.text import TfidfVectorizer
#We need this to convert the text data(mail data) to numerical forms, so that it can become meaningful
#If we send the text data into the ML model, the ML model can't understand it, so we are converting them into numericals

from sklearn.linear_model import LogisticRegression
#we need this model to train and classify whether mail is spam or ham

from sklearn.metrics import accuracy_score
#This is needed because we need to evaluate the model on the test data

#Data Collection and Pre-processing
#Loading the data from csv_file to a panda DataFrame

raw_mail_data=pd.read_csv('/Users/nissankararaojayanth/Spam Mail Detection/mail_data.csv')
#read_csv will load the data from csv file into pandas data frame

#Replace null values with a null string
mail_data=raw_mail_data.where((pd.notnull(raw_mail_data)),'')
#raw_mail_data is the variable name which we created before
#'where' is used to apply a condition, here in the raw_mail_data, if there is null values replace with null string

#printing the first 5 rows of data frame
print(mail_data.head())

#checking the number of rows and columns in the dataframe
#print(mail_data.shape)
#this will give the total rows(mails) and total labels(spam or ham)

#LABEL ENCODING
#Means labelling spam as 0 and ham as 1
#In the mail_data we will take the Category column and in that all the text data which is spam will be 0
#Labelling ham as 1
mail_data["Category"] = mail_data["Category"].map({
    "spam": 0,
    "ham": 1
})


#Separating the data as texts and label, because we need to feed the data separately to the model
#Text means the mail data, label means 0 or 1
#Text is x-axis, category is y-axis
#Here the input is text(mail data) that will be stored in X
#Here the output is label(0 or 1) that will be stored in Y

X = mail_data['Message']
Y = mail_data['Category']

# print(X)
# print(Y)

X_train, X_test, Y_train, Y_test=train_test_split(X,Y, test_size=0.2,random_state=3)
#training data message will go to X_train
#training data category will go to Y_train
#testing data message will go to X_test
#testing data category will go to X_train
#test_size means what amount of total data will be going to the testing data

# print(X.shape)
# print(X_train.shape)
# print(X_test.shape)
#Using this we can how many rows are in X(messages), training(messages) and testing(messages)



#Feature Extraction
#We will convert the text data(messages) into the numerical form, because if we feed the model with only text messages, it can't understand
feature_extraction = TfidfVectorizer(min_df=1, stop_words='english',lowercase=True)
#TfidfVectorizer will go through all the messages in the data and will give values to the words
#min_df means if the score for a particular word is less than 1, we should ignore it.
#stop_words is used to ignore the most common words which doesn't make any impact like does,is,was etc
#lowercase=True means we will convert all the words into lowercase

X_train_features= feature_extraction.fit_transform(X_train)
X_test_features= feature_extraction.transform(X_test)

#Convert Y_train and Y_test values as integers
#Even though the values are in 0,1 values they are considered  as dtype:object
Y_train=Y_train.astype('int')
Y_test=Y_test.astype('int')

#X_train means the data which is not yet converted, X_train_features means the data which is converted into features. Same for Y as well
# print(X_train_features)

#Initialize the model
model = LogisticRegression() 

 #Train the Model
model.fit(X_train_features,Y_train)

#Prediction on training data
#The values predicted by the model on the training data will be stored in the prediction_on_training_data
prediction_on_training_data = model.predict(X_train_features)
accuracy_on_training_data= accuracy_score(Y_train,prediction_on_training_data)
#here we will see the accuracy between Y_train(actual values in training data) and the values predicted by model (prediction_on_training_data)

#print('Accuracy on training data: ', accuracy_on_training_data)

#Prediction on test data
prediction_on_test_data = model.predict(X_test_features)
accuracy_on_test_data=accuracy_score(Y_test,prediction_on_test_data)

#print('Accuracy on test data: ', accuracy_on_test_data)

input_mail=["I've been searching for the right words to thank you for this breather. I promise i wont take your help for granted and will fulfil my promise. You have been wonderful and a blessing at all times."]

#convert text to feature vectors
input_data_features=feature_extraction.transform(input_mail)

#Making prediction
prediction=model.predict(input_data_features)
#print(prediction)

if prediction[0]==1:
 print("Ham Mail")

else:
 print("Spam Mail")
 