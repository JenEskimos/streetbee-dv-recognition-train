from mmdet.apis import init_detector, inference_detector, show_result
import mmcv
import numpy as np
from nanoid import generate


class Localizer():
    def __init__(self, config_file, checkpoint_file, confidence_th, clean_th=0.95):
        self.model = init_detector(config_file, checkpoint_file, device='cuda:0')
        self.confidence_th = confidence_th
        self.clean_th = clean_th

    def intersection(self, boxes1, boxes2):
        out = np.zeros(len(boxes1))
        condition = ~((boxes1[:,0]>boxes2[:,2])|(boxes2[:,0]>boxes1[:,2])|(boxes1[:,1]>boxes2[:,3])|(boxes2[:,1]>boxes1[:,3]))
        indexes = np.where(condition)[0]
        inter = np.vstack((np.maximum(boxes1[indexes,0], boxes2[indexes,0]),
                           np.maximum(boxes1[indexes,1], boxes2[indexes,1]),
                           np.minimum(boxes1[indexes,2], boxes2[indexes,2]),
                           np.minimum(boxes1[indexes, 3], boxes2[indexes, 3]))).T
        out[indexes] = self.area(inter)
        return out

    def area(self, boxes):
        return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    def clean(self, boxes, threshold):
        I = np.arange(len(boxes))
        I = np.ravel(np.broadcast_to(I, (len(boxes), len(boxes))).T)
        J = np.arange(len(boxes))
        J = np.ravel(np.broadcast_to(J, (len(boxes), len(boxes))))

        inter = self.intersection(boxes[I], boxes[J])
        inter = np.reshape(inter, (len(boxes), len(boxes)))
        inter = (inter.T / self.area(boxes)).T
        inter[range(inter.shape[0]),range(inter.shape[0])] = 0
        max = np.max(inter, axis=1)
        return np.where(max < threshold)[0]

    def __call__(self, img):
        res = inference_detector(self.model, img)[0]

        res = res[np.where(res[:, 4] > self.confidence_th)[0]]
        x1 = res[:, 0]
        x2 = res[:, 2]
        y1 = res[:, 1]
        y2 = res[:, 3]

        ind = self.clean(np.vstack((x1, y1, x2, y2)).T, self.clean_th)
        objects = [{'rect': [[int(y1[i]), int(x1[i])], [int(y2[i]), int(x2[i])]], 'type': 'sku', 'id': generate(),
                    'sku_id': None} for i in ind]
        return objects
