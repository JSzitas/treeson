import treeson
import pandas as pd

X_train = pd.read_parquet('data/X_train.parquet')
y_train = pd.read_parquet('data/y_train.parquet')
X_test = pd.read_parquet('data/X_test.parquet')


df = X_train.set_index(['id', 'moon']).join(y_train.set_index(['id', 'moon']))
df = df.head(10000)
df = df.astype('float')
df.reset_index(drop=True, inplace=True)
targets = []
i = 0
for i in range(len(df.columns)):
    if df.columns[i] in ['target_w','target_b','target_r','target_g']:
        targets.append(i)
    i += 1
max_depth = 10
min_nodesize = 200
seed = 42
n_tree = 100
df = [df[col].to_list() for col in df.columns]
print("Data")
model = treeson.MultitargetRandomForest(targets, max_depth, min_nodesize, seed)
model.fit_to_file(df, n_tree, "test_model", True, 1024)
print("Model")


df = X_test.drop(labels = ['moon', 'id'], axis = 1)
targets = [len(df.columns) + i for i in [0,1,2,3]]
df = df.astype('float')
df = [df[col].to_list() for col in df.columns]
print("Test")

model = treeson.MultitargetRandomForest(targets, max_depth, min_nodesize, seed)
pred = model.predict_from_file(df, "test_model", 0)
print("Predict")

print(pred)
