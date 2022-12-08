import pickle as pk
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv("Loan_Data.csv") 
df.loc[df["Loan_Status"] == 'N', "Loan_Status"] = 0
df.loc[df["Loan_Status"] == 'Y', "Loan_Status"] = 1 

feature_cols = ["Gender","Married","Dependents","Education","Self_Employed","ApplicantIncome","CoapplicantIncome","LoanAmount","Loan_Amount_Term","Credit_History", "Property_Area"]
label_encoder_map = {}


df_copy = df.applymap(lambda s:s.lower() if type(s) == str else s)
df_encoded_labels = df_copy.copy()


for i in range(len(feature_cols)):
    labels = df_encoded_labels[feature_cols[i]].astype('category').cat.categories.tolist()
    replace_map_comp = {feature_cols[i] : {k :v for k, v in zip( labels, list(range(0, len(labels))))}}

    label_encoder_map.update(replace_map_comp)
    df_encoded_labels.replace(replace_map_comp, inplace=True)

pk.dump(label_encoder_map, open("label_encoder_map.pkl", "wb"))

df_encoded_labels.isnull().sum()
df_encoded_labels = df_encoded_labels.fillna(df_encoded_labels.mean())

X = df_encoded_labels.loc[:, feature_cols]
y = df_encoded_labels.loc[:, 'Loan_Status']
df_encoded_labels_copy = df_encoded_labels

df_encoded_labels = df_encoded_labels.drop('Loan_Status', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

classifier = LogisticRegression(random_state = 0, solver='lbfgs', multi_class='auto', max_iter=1000)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
score =accuracy_score(y_test,y_pred)
print("Accuracy Score: ", score)

pk.dump(classifier, open("model.pkl", "wb")) 

X_col = df_encoded_labels_copy.loc[:, feature_cols]
y_col = df_encoded_labels_copy.loc[:, 'Loan_Status'] 
  
df_encoded_labels_copy['Probability of Loan Approval'] = classifier.predict_proba(df_encoded_labels[X_col.columns])[:,1] 
print( df_encoded_labels_copy[['Loan_Status', 'Probability of Loan Approval']])

probs_y=classifier.predict_proba(X_test)
probs_y = np.round(probs_y, 2)

df_encoded_labels_copy['Probability of Loan Approval'] = classifier.predict_proba(df_encoded_labels_copy[X.columns])[:,1] 


