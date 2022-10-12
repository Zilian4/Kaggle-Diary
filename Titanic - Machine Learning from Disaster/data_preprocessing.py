# data preprocessing
# 1.handle empty data
# 2.encoding words
# 3.feature selection
# 4.normalization

import pandas as pd
from sklearn.model_selection import train_test_split


def getTitle(name):
    str1 = name.split(',')[1]  # Mr. Owen Harris
    str2 = str1.split('.')[0]  # Mr
    # strip() 方法用于移除字符串头尾指定的字符（默认为空格）
    str3 = str2.strip()
    return str3


class Dataloader:
    def __init__(self):
        train = pd.read_csv(r'./data/train.csv')
        test = pd.read_csv(r'./data/test.csv')
        # For training and testing:(891, 12) For prediction: (418, 11)
        # print(train.shape,test.shape)

        data = train.append(test, ignore_index=True)
        # data:[1309 rows x 12 columns]

        data['Fare'] = data['Fare'].fillna(data['Fare'].mean())
        data['Age'] = data['Age'].fillna(data['Age'].mean())

        data['Embarked'] = data['Embarked'].fillna('S')
        #
        # S    914
        # C    270
        # Q    123
        # print(pd.value_counts(data['Embarked']))
        data = data.drop(['Cabin', 'Ticket'], axis=1)
        # female <--0 and male <--1
        sex_mapDict = {'female': 0, "male": 1}
        data["Sex"] = data['Sex'].map(sex_mapDict)

        embarked = data['Embarked']
        # pd.get_dummies: Encoding with one-hot
        embarked = pd.get_dummies(embarked, prefix='Embarked')
        data = pd.concat([data, embarked], axis=1)
        # delete colum 'Embarked'
        data.drop('Embarked', axis=1, inplace=True)

        # do the same things to Pclass
        Pclass = data['Pclass']
        Pclass = pd.get_dummies(Pclass, prefix='Pclass')
        data = pd.concat([data, Pclass], axis=1)
        data.drop('Pclass', axis=1, inplace=True)

        # name : what we focus on is the title

        # get title
        title = pd.DataFrame()
        title['title'] = data['Name'].map(getTitle)
        # print(title.value_counts()) show the classes of title

        title_mapdic = {"Mr": 'Mr',
                        'Miss': "Miss",
                        'Mrs': 'Mrs',
                        'Ms': 'Mrs',
                        'Master': 'Masters',
                        'Rev': 'Officer',
                        'Dr': 'Officer',
                        'Jonkheer': 'Officer',
                        'Capt': 'Officer',
                        'Mme': 'Mrs',
                        'the Countess': 'Royalty',
                        'Col': 'Officer',
                        'Major': 'Officer',
                        'Dona': 'Royalty',
                        'Don': 'Royalty',
                        'Lady': 'Royalty',
                        'Sir': 'Royalty'}

        title['title'] = title['title'].map(title_mapdic)

        title = pd.get_dummies(title, prefix='Title')
        data = pd.concat([data, title], axis=1)
        data = data.drop({'Name', 'PassengerId'}, axis=1)
        data_X = data.loc[0:890, :]
        self.data_X = data_X.drop('Survived', axis=1)
        self.data_Y = data.loc[0:890, "Survived"]
        self.data_predict = data.loc[891:, :].drop('Survived', axis=1)

    def get_train_data(self):
        return train_test_split(self.data_X, self.data_Y, train_size=0.9, test_size=0.1, shuffle=True, random_state=5)

    def get_predict_data(self):
        return self.data_predict
