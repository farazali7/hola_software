from sklearn.svm import SVC
import torch
import pickle
import pywt

with open('jetson_nano/release_models/train_data_svc_2.pkl', 'rb') as r:
    train_x, train_y = pickle.load(r)

print(f"train_x: {train_x.shape}")
model = SVC(probability=True)
model.fit(train_x, train_y)


with open('jetson_nano/release_models/svc_3.pkl', 'wb') as f:
    pickle.dump(model, f)
