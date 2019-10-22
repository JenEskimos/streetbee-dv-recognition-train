import urllib.request, json
from PIL import Image
import glob
import os
import io
import copy
import numpy as np
from math import *
import random
import multiprocessing


def getmatrix(angle):
    return np.array([[cos(angle * np.pi / 2.0), -sin(angle * np.pi / 2.0)],
                     [sin(angle * np.pi / 2.0), cos(angle * np.pi / 2.0)]])


def rotate_rects(rect, old_center, new_center, angle):
    rect = np.array(rect) - old_center
    return np.dot(getmatrix(angle), rect.T).T + new_center


class Loader:
    def __init__(self, category):
        self.category = category

    def __call__(self, file):
        print(file)
        with open(file) as fp:
            data = json.load(fp)
        file_id = os.path.split(file)[-1].split('.')[0]
        img_link = data["image_url"]
        with urllib.request.urlopen(img_link) as fd:
            image_file = io.BytesIO(fd.read())
            im = Image.open(image_file)
        old_center = im.size[1] * 0.5, im.size[0] * 0.5

        if random.random() < 0.1:
            split = 'test'
        else:
            split = 'train'

        for rotation in range(1):
            im_rotate = im.rotate(90 * rotation, expand=True)
            new_center = im_rotate.size[1] * 0.5, im_rotate.size[0] * 0.5
            new_data = copy.deepcopy(data)
            for o in new_data['objects']:
                if 'rect' in o:
                    key = 'rect'
                else:
                    if 'polygon' in o:
                        key = 'polygon'
                    else:
                        continue
                o[key] = rotate_rects(o[key], old_center, new_center, rotation).tolist()

            out = os.path.join('/Downloads', self.category, split)
            if not os.path.exists(out):
                os.makedirs(out)

            with open(os.path.join(out, file_id + '_%s.json' % rotation), 'w') as file:
                json.dump(new_data, file)
            im_rotate.save(os.path.join(out, file_id + '_%s.jpg' % rotation), 'jpeg')


if __name__ == '__main__':
    categorys = ['']
    for category in categorys:
        files = []
        for file in glob.glob(os.path.join(category, '*.json')):
            files.append(file)
        p = multiprocessing.Pool(6)
        p.map(Loader(category), files)
        p.close()