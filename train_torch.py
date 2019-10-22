import h5py
import data_sampler
import torch_transformer
import numpy as np
import pickle

train = h5py.File('data/train.hdf5')
test = h5py.File('data/test.hdf5')
data = data_sampler.DataSampler(train, test, new_sku='new')
with open('label_encoder.p', 'wb') as file:
    pickle.dump(data.le, file)

model = torch_transformer.Classifier(len(data.le.classes_))
model.fit(data, 300000)

train_scores = np.array([model.score([data.get_train_scene(id)]) for id in data.ids['train']])
print('mean train score: ', np.mean(train_scores), np.percentile(train_scores, 20), np.percentile(train_scores, 50), np.percentile(train_scores, 80))

test_scores = np.array([model.score([data.get_test_scene(id)]) for id in data.ids['test']])
print('mean test score: ', np.mean(test_scores), np.percentile(test_scores, 20), np.percentile(test_scores, 50), np.percentile(test_scores, 80))
