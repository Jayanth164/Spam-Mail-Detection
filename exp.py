import numpy as np
import pandas as pd
#We use pandas to create the data frames, because the data which we have is in csv format and it is not easy to put that data into structured format
#So we will use pandas to convert that data to data frame


from sklearn.model_selection import train_test_split
#we need this, because we need to split the data into training and testing data


from sklearn.feature_extraction.text import TfidfVectorizer
#We need this to convert the text data(mail data) to numerical forms, so that it can become meaningful
#If we send the text data into the ML model, the ML model can't understand it, so we are converting them into numericals


from sklearn.linear_model import LogisticRegression
#we need this model to train and classify whether mail is spam or ham


from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
#This is needed because we need to evaluate the model on the test data


#Data Collection and Pre-processing
#Loading the data from csv_file to a panda DataFrame


raw_mail_data = pd.read_csv(
    "/Users/nissankararaojayanth/Spam Mail Detection/mail_data.csv"
)
#read_csv will load the data from csv file into pandas data frame


#Validate that the expected columns are available
required_columns = {"Category", "Message"}

if not required_columns.issubset(raw_mail_data.columns):
    missing_columns = required_columns - set(raw_mail_data.columns)
    raise ValueError(f"Missing required columns: {missing_columns}")


#Replace null values with a null string
#Only the Message column is text data, so we fill missing messages with an empty string
mail_data = raw_mail_data.copy()
mail_data["Message"] = mail_data["Message"].fillna("")
#raw_mail_data is the variable name which we created before
#'where' is used to apply a condition, here in the raw_mail_data, if there is null values replace with null string


#printing the first 5 rows of data frame
print(mail_data.head())


#checking the number of rows and columns in the dataframe
#print(mail_data.shape)
#this will give the total rows(mails) and total labels(spam or ham)


#Remove rows that do not have a category
mail_data = mail_data.dropna(subset=["Category"])


#Check whether unexpected category values are present
valid_categories = {"spam", "ham"}
unexpected_categories = set(mail_data["Category"].unique()) - valid_categories

if unexpected_categories:
    raise ValueError(
        f"Unexpected category values found: {unexpected_categories}"
    )


#LABEL ENCODING
#Means labelling spam as 0 and ham as 1
#In the mail_data we will take the Category column and in that all the text data which is spam will be 0
#Labelling ham as 1
mail_data["Category"] = mail_data["Category"].map({
    "spam": 0,
    "ham": 1
})


#Check that label encoding did not create missing values
if mail_data["Category"].isna().any():
    raise ValueError("Some categories could not be encoded")


#Convert target labels to integers
#Even though the values are in 0,1 values they are considered as dtype:object
mail_data["Category"] = mail_data["Category"].astype(int)


#Separating the data as texts and label, because we need to feed the data separately to the model
#Text means the mail data, label means 0 or 1
#Text is x-axis, category is y-axis
#Here the input is text(mail data) that will be stored in X
#Here the output is label(0 or 1) that will be stored in Y


X = mail_data["Message"]
Y = mail_data["Category"]


# print(X)
# print(Y)


X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=3,
    stratify=Y
)
#training data message will go to X_train
#training data category will go to Y_train
#testing data message will go to X_test
#testing data category will go to Y_test
# test_size=0.2 means 20% of the data is used for testing and the remaining 80% is used for training.


# print(X.shape)
# print(X_train.shape)
# print(X_test.shape)
#Using this we can how many rows are in X(messages), training(messages) and testing(messages)


#Feature Extraction
#We will convert the text data(messages) into the numerical form, because if we feed the model with only text messages, it can't understand
feature_extraction = TfidfVectorizer(
    min_df=1,
    stop_words="english",
    lowercase=True
)
#TfidfVectorizer learns a vocabulary from the training messages and converts each message into numerical TF-IDF feature values.
#min_df specifies the minimum number of documents in which a term must appear. With min_df=1, a term must appear in at least one document.
#if we put min_df =2 then we will ignore the words which appears only in 1 document
# stop_words="english" removes common English words such as "is", "was", and "the". These words may provide limited value for distinguishing spam from ham.
#lowercase=True means we will convert all the words into lowercase


#Fit the vectorizer only on training data
#This prevents information from the test data leaking into the training process
X_train_features = feature_extraction.fit_transform(X_train)

#Use the vocabulary learned from the training data to transform test data
X_test_features = feature_extraction.transform(X_test)


#X_train means the data which is not yet converted, X_train_features means the data which is converted into features. Same for Y as well
# print(X_train_features)


#Initialize the model
model = LogisticRegression(
    max_iter=1000
)


# Train the model using the numerical training features and their correct labels.
model.fit(X_train_features, Y_train)


#Prediction on training data
#The values predicted by the model on the training data will be stored in the prediction_on_training_data
prediction_on_training_data = model.predict(X_train_features)

accuracy_on_training_data = accuracy_score(
    Y_train,
    prediction_on_training_data
)
#here we will see the accuracy between Y_train(actual values in training data) and the values predicted by the model (prediction_on_training_data)


print(
    "Accuracy on training data: ",
    accuracy_on_training_data
)


#Prediction on test data
prediction_on_test_data = model.predict(X_test_features)

accuracy_on_test_data = accuracy_score(
    Y_test,
    prediction_on_test_data
)


print(
    "Accuracy on test data: ",
    accuracy_on_test_data
)


#Print detailed test-data metrics
#Accuracy alone may not be enough for a spam classification problem
# This report also displays precision, recall, F1-score, and support.
print("\nClassification report:")
print(
    classification_report(
        Y_test,
        prediction_on_test_data,
        target_names=["Spam", "Ham"]
    )
)


#Print the confusion matrix
# With labels [0, 1]:
# row 0 = actual spam, row 1 = actual ham
# column 0 = predicted spam, column 1 = predicted ham
print("Confusion matrix:")
print(confusion_matrix(Y_test, prediction_on_test_data))


input_mail = [
    "I've been searching for the right words to thank you for this breather. "
    "I promise i wont take your help for granted and will fulfil my promise. "
    "You have been wonderful and a blessing at all times."
]


# Convert the new message into TF-IDF features
# using the vocabulary learned from the training data.
# Predict whether the new message is spam or ham.
input_data_features = feature_extraction.transform(input_mail)


#Making prediction
prediction = model.predict(input_data_features)
#print(prediction)


if prediction[0] == 1:
    print("Ham Mail")


else:
    print("Spam Mail")



# 1. Loading the data from a csv file to the Pandas dataframe
# 2. Validate that the required columns are present or not, if not present raise error
# 3. Replace null values with a null string 
# 4. Remove the rows that don't have any category 
# 5. Check whether any other values are present in category other than the expected, if yes raise them
# 6. Now Label encoding, label the categories in numerical forms as 0 or 1
# 7. Check whether all the categories are encoded or not, if not raise error
# 8. Convert labels to integers, even though they are in numerical form, they are considered as objects
# 9. Separate the data as text and label as we need to send them separately to the model
#10. Text (mail) will be stored in a variable (X) and category will be stored in another variable (Y)
#11. Now split the data into training and testing data, test_size will be around 0.2 that means around 20%
#12. Now we should do the feature extraction by converting the data (mail) into numerical form, because we can't feed model only with text data
#13. Now train the model on the training data and test it on training and test data



