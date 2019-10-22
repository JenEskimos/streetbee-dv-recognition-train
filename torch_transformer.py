from math import *
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from torch.autograd import Variable
from encoder import resnet50
import time
import os


class MultiHeadAttention(nn.Module):
    def __init__(self, input_size, position_code_size, key_size, value_size, n_head, out_size, resedual=False,
                 dropout=0.5):
        super(MultiHeadAttention, self).__init__()
        self.input_size = input_size
        self.dropout = dropout
        self.resedual = resedual
        self.key_size = key_size
        self.value_size = value_size
        self.n_head = n_head
        self.out_size = out_size

        self.W_k = nn.Linear(input_size + position_code_size, key_size * n_head, bias=False)
        self.W_q = nn.Linear(input_size + position_code_size, key_size * n_head, bias=False)
        self.W_v = nn.Linear(input_size + position_code_size, value_size * n_head, bias=False)

        self.W_out1 = nn.Linear(value_size * n_head, out_size)
        self.W_out1 = nn.Linear(value_size * n_head, out_size)
        self.W_out2 = nn.Linear(out_size, out_size)
        self.bn1 = nn.BatchNorm1d(out_size)
        self.bn2 = nn.BatchNorm1d(out_size)

        self.relu = nn.ReLU(inplace=True)
        self.drop_layer = torch.nn.Dropout(dropout)

    def transform(self, X):
        Q = self.W_q(X)
        K = self.W_k(X)
        V = self.W_v(X)

        #Q = torch.reshape(Q, (X.shape[0], self.n_head, self.key_size, Q.shape[2]))
        #K = torch.reshape(K, (X.shape[0], self.n_head, self.key_size, K.shape[2]))
        #V = torch.reshape(V, (X.shape[0], self.n_head, self.value_size, V.shape[2]))

        Q = torch.reshape(Q, (X.shape[0], self.n_head, self.key_size)).transpose(0, 1)
        K = torch.reshape(K, (X.shape[0], self.n_head, self.key_size)).transpose(0, 1)
        V = torch.reshape(V, (X.shape[0], self.n_head, self.value_size)).transpose(0, 1)

        A = torch.bmm(Q, K.transpose(1, 2)) / sqrt(self.key_size)
        A = torch.softmax(A, dim=2)
        Z = torch.bmm(A, V)

        Z = Z.transpose(0, 1)
        Z = torch.reshape(Z, (X.shape[0], self.n_head * self.value_size))

        return Z

    def agregate(self, Z, skip):
        Z = self.W_out1(Z)
        Z = self.bn1(Z)
        Z = self.relu(Z)
        Z = self.drop_layer(Z)
        Z = self.W_out2(Z)
        Z = self.relu(self.bn2(Z) + skip)
        return Z

    def __call__(self, emb, boxes):
        out = []
        skip = []

        for x, b in zip(emb, boxes):
            X = torch.cat((x, b), dim=1)
            Z = self.transform(X)
            out.append(Z)
            skip.append(x)

        skip = torch.cat(skip, dim=0)
        Z = torch.cat(out, dim=0)
        Z = self.agregate(Z, skip)

        out = []
        k = 0
        for x in emb:
            out.append(Z[k: k + len(x)])
            k += len(x)

        return out


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.resnet = resnet50()

    def forward(self, x):
        return self.resnet(x)


class TransformerModel(nn.Module):
    def __init__(self, n_label):
        super(TransformerModel, self).__init__()
        self.encoder = Encoder()
        self.projection_box = nn.Linear(8, 32)
        self.projection_box_bn = nn.BatchNorm1d(32)
        self.box_transformer1 = MultiHeadAttention(100, 32, 32, 100, 5, 100, True)
        self.box_transformer2 = MultiHeadAttention(100, 32, 32, 100, 5, 100, True)

        # transformer2=MultiHeadAttention(100, 25, 100, 4, 100, True)

        self.final = nn.Linear(100, n_label)
        #self.final_pretrain = nn.Linear(100, n_label)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, data):
        x = torch.cat([row[0] for row in data], dim=0)
        x = self.encoder(x)

        boxes = torch.cat([row[1] for row in data], dim=0)
        boxes = torch.reshape(boxes, (boxes.shape[0], -1))

        boxes = self.projection_box(boxes)
        boxes = self.projection_box_bn(boxes)

        centers = []
        emb = []
        k = 0
        for i in range(len(data)):
            emb.append(x[k: k + len(data[i][0])])
            centers.append(boxes[k: k + len(data[i][0])])
            k += len(data[i][0])

        emb = self.box_transformer1(emb, centers)
        emb = self.box_transformer2(emb, centers)

        out = torch.cat(emb, dim=0)

        # out = F.dropout(out)
        out = self.final(out)

        return out


def np_to_tensor(x, use_gpu=True):
    if use_gpu:
        return Variable(torch.from_numpy(x).cuda())
    else:
        return Variable(torch.from_numpy(x))


class Classifier():
    def __init__(self, n_classes, use_gpu=True):
        self.model = TransformerModel(n_classes)
        self.use_gpu = use_gpu
        if use_gpu:
            self.model.cuda()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()

    def save_log(self, folder, step):
        if not os.path.exists(folder):
            os.makedirs(folder)
        np.save(os.path.join(folder, f'lcurve_{step}.npy'), self.lcurve)
        torch.save(self.model.state_dict(), os.path.join(folder, f'model_{step}.pth'))

    def drop_learning_rate(self, alpha=0.1):
        for g in self.optimizer.param_groups:
            g['lr'] *= alpha

    def predict_scenes(self, X):
        out = []
        self.model.eval()
        for i in range(0, len(X)):
            with torch.no_grad():
                x = [[np_to_tensor(X[i][0], self.use_gpu), np_to_tensor(X[i][1], self.use_gpu)]]
                result = self.model(x)
                if self.use_gpu:
                    result = result.cpu()
            out.append(np.argmax(result.numpy(), axis=1))
        return out

    def predict_emb(self, imgs_np, batch_size=200):
        self.model.eval()
        result = []
        with torch.no_grad():
            for i in range(0, imgs_np.shape[0], batch_size):
                out = self.model.encoder(np_to_tensor(imgs_np))
                if self.use_gpu:
                    out = out.cpu()
                result.append(out.numpy())
        return np.concatenate(result, axis=0)

    def score(self, data):
        Y = [row[2] for row in data]
        pr = self.predict_scenes([row[:2] for row in data])
        acc = []
        for predicted, true in zip(pr, Y):
            acc.append(accuracy_score(true, predicted))
        return np.mean(acc)

    def fit(self, data, n_iter=300000, n_scene=3, max_image=200, drop_steps=((250000, 0.1),), snapshot_step=50000,
            log_folder='model_log'):
        self.lcurve = []
        self.model.train()
        drop_steps = {row[0]: row[1] for row in drop_steps}

        for step in range(n_iter):
            if step in drop_steps:
                self.drop_learning_rate(drop_steps[step])
            if step % snapshot_step == 0:
                self.save_log(log_folder, step)
            start = time.time()

            batch = data.get_train_batch(n_scene, max_image)
            if len(batch) == 0:
                continue
            X = []
            Y = []
            W = []
            for i in range(len(batch)):
                X.append([np_to_tensor(batch[i][0], self.use_gpu), np_to_tensor(batch[i][1], self.use_gpu)])
                u, c = np.unique(batch[i][2], return_counts=True)
                new_w = np.zeros(len(batch[i][2]), dtype=np.float32)
                for l, count in zip(u, c):
                    new_w[np.where(batch[i][2] == l)[0]] = 1. / (count * len(u))

                Y.append(batch[i][2])
                W.append(new_w)
            Y = torch.tensor(np.concatenate(Y, axis=0), dtype=torch.long)
            if self.use_gpu:
                Y = Y.cuda()
            W = np_to_tensor(np.concatenate(W, axis=0).astype(np.float32) / len(batch), self.use_gpu)

            self.optimizer.zero_grad()
            predict = self.model(X)
            loss = torch.nn.functional.cross_entropy(predict, Y, reduction='none')
            loss = torch.sum(loss * W)
            loss.backward()
            self.optimizer.step()

            self.lcurve.append(float(loss.item()))

            print('iter: %s' % step, 'time: %s' % (time.time() - start), 'loss %s' % np.mean(self.lcurve[-10:]))

        self.save_log(log_folder, 'final')
        self.model.eval()
