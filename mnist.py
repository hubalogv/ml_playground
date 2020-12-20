import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


train = pd.read_csv(r"C:\_ws\datasets\mnist\train.csv")
X_test = pd.read_csv(r"C:\_ws\datasets\mnist\test.csv")

Y_train = train["label"]

# Drop 'label' column
X_train = train.drop(labels=["label"], axis=1)

random_seed = 2

X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.1, random_state=random_seed)
assert isinstance(Y_val, pd.Series)

model = LogisticRegression(max_iter=50)

model.fit(X_train, Y_train)

predictions = model.predict(X_val)

print('score: ', accuracy_score(Y_val, predictions))

results = model.predict(X_test)

output = pd.DataFrame({'ImageId': range(1,len(results)+1),
                       'Label': results})
output.to_csv('submission.csv', index=False)
