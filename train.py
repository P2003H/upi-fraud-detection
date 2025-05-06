import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib


df = pd.read_csv("PS_20174392719_1491204439457_log.csv")
df = df[df['type'].isin(['TRANSFER', 'PAYMENT'])]


df['hour'] = df['step'] % 24
df['is_night'] = df['hour'].apply(lambda x: 1 if x < 6 else 0)
df['amount_ratio'] = df['amount'] / (df['oldbalanceOrg'] + 1)
df['sender_balance_change'] = df['oldbalanceOrg'] - df['newbalanceOrig']
df['receiver_balance_change'] = df['newbalanceDest'] - df['oldbalanceDest']
df['orig_balance_zero'] = (df['oldbalanceOrg'] == 0).astype(int)
df['dest_balance_zero'] = (df['oldbalanceDest'] == 0).astype(int)
df = pd.get_dummies(df, columns=['type'], drop_first=True)


df.drop(['nameOrig', 'nameDest', 'isFlaggedFraud'], axis=1, inplace=True)


fraud = df[df['isFraud'] == 1]
nonfraud = df[df['isFraud'] == 0].sample(len(fraud)*3, random_state=42)
balanced_df = pd.concat([fraud, nonfraud])


X = balanced_df.drop('isFraud', axis=1)
y = balanced_df['isFraud']


model = RandomForestClassifier(n_estimators=100, class_weight={0:1, 1:10}, random_state=42)
model.fit(X, y)


joblib.dump(model, "fraud_rf_model.joblib")
print("Model trained and saved!")
