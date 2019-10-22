import h5py
import numpy as np
from sklearn.preprocessing import LabelEncoder
from scipy.optimize import nnls
from matplotlib import pyplot as pl
from PIL import Image, ImageEnhance
import cupy
from math import *


def augument(x):
    angles = np.random.choice([0, 1, 2, 3], len(x))
    mirrow = np.random.uniform(0, 1, len(x))
    for k in [1, 2, 3]:
        indexes = np.where(angles == k)[0]
        x[indexes] = np.rot90(x[indexes], k, (2, 3))

    x[np.where(mirrow > 0.5)[0], :, :, :] = x[np.where(mirrow > 0.5)[0], :, ::-1, :]


def color_augment(p):
    br = ImageEnhance.Brightness(p)
    randomval = np.random.normal(1, 0.2, 3)
    p = br.enhance(randomval[0])
    ct = ImageEnhance.Contrast(p)
    p = ct.enhance(randomval[1])
    ct = ImageEnhance.Color(p)
    p = ct.enhance(randomval[2])
    return p


class DataSampler:
    def __init__(self, train, test, count_threshold=10, k=0.25, size=112, new_sku='trash'):
        self.train = train
        self.test = test

        self.ids = {'test': list(self.test.keys()), 'train': list(self.train.keys())}
        self.k = k
        self.size = size

        self.train_skus = []
        self.train_ids = []
        for key in self.train.keys():
            self.train_ids.append(key)
            self.train_skus += [bytes.decode(x) for x in self.train[key]['skus'][:]]

        self.test_skus = []
        for key in self.test.keys():
            self.test_skus += [bytes.decode(x) for x in self.test[key]['skus'][:]]

        np.save('train_skus.npy',np.unique(self.train_skus))
        np.save('test_skus.npy', np.unique(self.test_skus))

        self.train_skus, counts = np.unique(self.train_skus, return_counts=True)
        self.replace_to_trash = {key: new_sku for key in self.train_skus[np.where(counts <= count_threshold)[0]]}
        print('new sku: ', len(self.replace_to_trash))
        print('new sku: ', len(self.train_skus))
        self.train_skus = self.train_skus[np.where(counts > count_threshold)[0]]
        self.train_skus = np.unique(self.train_skus.tolist() + ['trash', new_sku])

        self.le = LabelEncoder()
        self.le.fit(self.train_skus)

        self.id_probs = np.zeros((len(self.train_ids), len(self.train_skus)), np.float)
        for i, key in enumerate(self.train_ids):
            skus = np.array([self.replace_to_trash.get(bytes.decode(x), bytes.decode(x))
                             for x in self.train[key]['skus'][:]
                             if self.replace_to_trash.get(bytes.decode(x), bytes.decode(x)) in self.train_skus])
            skus = self.le.transform(skus)
            skus, p = np.unique(skus, return_counts=True)
            p = p / np.sum(p)
            self.id_probs[i, skus] = p

        self.id_probs = self.id_probs / np.sum(self.id_probs , axis=0)
        print(self.le.classes_)

    def get_train_ids(self, n_scene):
        ind = np.random.choice(len(self.le.classes_), n_scene)
        ids = []
        for i in ind:
            ids += np.random.choice(self.train_ids, 1, p=self.id_probs[:, i]).tolist()
        return ids

    def resize_images(self, images, augment=True):
        if augment:
            crop = np.random.uniform(0, 1, (len(images), 4))
        else:
            crop = np.full((len(images), 4), 0.5)
        offset = 2 * self.k / (1 + 2 * self.k)
        size = images.shape[2]
        out = []
        for i in range(len(images)):
            left = crop[i, :2] * offset * size
            right = size - crop[i, 2:] * offset * size
            new_img = Image.fromarray(images[i].T).crop([left[0], left[1], right[0], right[1]])\
                .resize((self.size, self.size))
            if augment:
                new_img = color_augment(new_img)
            new_img = (2 * np.array(new_img).T / 255.0 - 1).astype(np.float32)
            out.append(new_img)

        out = np.array(out)
        if augment:
            augument(out)
        return out

    def augment_centers(self, centers):
        shape = centers.shape
        centers = centers.reshape((-1, 2))
        angle = np.random.uniform(0, 2 * np.pi, 1)[0]
        scale = np.random.uniform(0.5, 2, 1)[0]
        rotation = np.array([[cos(angle), - sin(angle)], [sin(angle), cos(angle)]])
        mirrow = np.random.choice([-1, 1], 1)[0]
        mirrow = np.array([[mirrow * scale, 0], [0, scale]])

        transforms = np.dot(rotation, mirrow)
        centers = np.dot(transforms, centers.T).T
        centers = centers.reshape(shape)
        centers += np.random.normal(0, 0.001, centers.shape)
        return centers.astype(np.float32)

    def get_train_batch(self, n_scene, max_sku=300, drop_rate=0.2, mask_rate=0.1):
        ids = self.get_train_ids(n_scene)
        out = []
        total = 0
        for id in ids:
            skus = [self.replace_to_trash.get(bytes.decode(sku), bytes.decode(sku)) for sku in self.train[id]['skus'][:]]
            ind = np.where([sku in self.le.classes_ for sku in skus])[0]

            while True:
                sku_drop = np.random.uniform(0, 1, len(ind))
                new_ind = ind[np.where(sku_drop < 1 - drop_rate)[0]]
                if len(new_ind) > 0:
                    ind = new_ind
                    break

            skus = self.le.transform(np.array(skus)[ind]).astype(np.int32)

            faces = self.train[id]['faces'][:]
            faces = self.resize_images(faces[ind], True)

            centers = self.augment_centers(self.train[id]['centers'][:][ind])

            if total + len(skus) > max_sku:
                if max_sku - total > 0:
                    ind = np.random.choice(len(skus), max_sku - total, replace=False)
                    out.append([faces[ind], centers[ind], skus[ind]])
                return out
            total += len(skus)

            if len(faces) > 1 and mask_rate is not None:
                mask_ind = np.where(np.random.uniform(0, 1, len(faces)) < mask_rate)[0]
                faces[mask_ind] = np.random.uniform(-1, 1, faces[mask_ind].shape)

            out.append([faces, centers, skus])
        return out

    def get_train_scene(self, id):
        skus = [self.replace_to_trash.get(bytes.decode(sku), bytes.decode(sku)) for sku in self.train[id]['skus'][:]]
        ind = np.where([sku in self.le.classes_ for sku in skus])[0]
        skus = self.le.transform(np.array(skus)[ind]).astype(np.int32)

        faces = self.train[id]['faces'][:]
        faces = self.resize_images(faces[ind], False)

        centers = self.train[id]['centers'][:][ind].astype(np.float32)
        return [faces, centers, skus]

    def get_test_scene(self, id):
        skus = [self.replace_to_trash.get(bytes.decode(sku), bytes.decode(sku)) for sku in self.test[id]['skus'][:]]
        ind = np.where([sku in self.le.classes_ for sku in skus])[0]
        if len(ind) > 0:
            skus = self.le.transform(np.array(skus)[ind]).astype(np.int32)

            faces = self.test[id]['faces'][:]
            faces = self.resize_images(faces[ind], False)

            centers = self.test[id]['centers'][:][ind].astype(np.float32)
            return [faces, centers, skus]
        return None

