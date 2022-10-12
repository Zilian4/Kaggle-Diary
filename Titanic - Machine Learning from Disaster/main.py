from data_preprocessing import Dataloader
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

D = Dataloader()
x_train, x_test, y_train, y_test = D.get_train_data()

# LogisticRegression
model1 = LogisticRegression(solver='liblinear')
model1.fit(X=x_train, y=y_train)
# score_logistic = model1.score(x_test, y_test)

# RandomForestClassifier
model2 = RandomForestClassifier()
model2.fit(X=x_train, y=y_train)
score_RFC = model2.score(x_test, y_test)

# Get data for prediction and make prediction
data_pre = D.get_predict_data()

prediction = model1.predict(data_pre).astype(int)
gender_submission = pd.read_csv('./data/test.csv')
gender_submission.insert(1, "Survived", prediction)
A = gender_submission[{'PassengerId', "Survived"}]
A.to_csv('./gender_submission_LogisticRegression.csv', index=False)

