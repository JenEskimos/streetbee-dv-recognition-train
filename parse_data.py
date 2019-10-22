import json
import os,glob
from PIL import Image
import numpy as np
import pickle
import h5py
from localization import Localizer
import pandas

skip_sku={'94579e26-f3d7-429a-9c88-de79dd780686',
          '5ddd5fd0-1c69-42ae-9e71-fd84d561696d',
          '5e42a412-dc2f-4aa1-9beb-b707e6480d68',
          '3540037a-c9b8-403a-b568-37105e8bfadb',
          '568a75a6-8063-4fb4-8123-4eef1c3b0738',
          'bcb0f7ab-6753-4dcd-b8b5-11c19911d8dc',
          '12908f9a-6a5f-425a-a802-7108183c8583',
          'f00bfe6a-dd8e-4e63-baca-e833733a84fb',
          '303df774-4ef4-4f07-91a3-a37cfe6155f8',
          '3540037a-c9b8-403a-b568-37105e8bfadb',}

sku_matching = pandas.read_csv('doubles.e2e.csv').as_matrix()
sku_matching = {row[0]: row[1] for row in sku_matching}


workdir = '/app'
category = ''
out_folder = 'data'
size = 112
k = 0.25



def load_localizer(params='/app/params'):
    src = os.path.join(params, 'params_localization')
    with open(os.path.join(src, 'post_config.json')) as file:
        data = json.load(file)
    localizer = Localizer(os.path.join(src, 'config.py'), os.path.join(src, 'weights.pth'), data['confidence_th'])
    return localizer


def reformat_rect(rect):
    rect = np.array(rect)
    x1, x2 = min(rect[:, 0]), max(rect[:, 0])
    y1, y2 = min(rect[:, 1]), max(rect[:, 1])
    return [(x1,y1),(x2,y2)]


def area(rect):
    return abs(rect[0][0] - rect[1][0]) * abs(rect[0][1] - rect[1][1])


def intersection_area(rect1,rect2):
    if rect2[0][1] > rect1[1][1] or rect2[0][0] > rect1[1][0] \
            or rect1[0][1] > rect2[1][1] or rect1[0][0] > rect2[1][0]:
        return 0
    else:
        p1 = (max(rect1[0][0], rect2[0][0]), max(rect1[0][1], rect2[0][1]))
        p2 = (min(rect1[1][0], rect2[1][0]), min(rect1[1][1], rect2[1][1]))
        return area((p1, p2))


def add_trash_box(img, objects, localizer, thresold=0.5):
    loc_obj = localizer(np.array(img))
    add_obj = []
    for p_obj in loc_obj:
        row = []
        for r_obj in objects:
            if 'rect' in r_obj:
                r_rect = reformat_rect(r_obj['rect'])
            else:
                r_rect = reformat_rect(r_obj['polygon'])
            intersect = intersection_area(reformat_rect(p_obj['rect']), r_rect)
            union = area(reformat_rect(p_obj['rect']))+area(reformat_rect(r_rect)) - intersect
            row.append(intersect / union)
        if np.max(row) < thresold:
            add_obj.append(p_obj)
            add_obj[-1]['sku_id'] = 'trash'
    print(len(add_obj))
    return add_obj


def parse_data(data_path, out_path, localizer):
    print(data_path)
    with h5py.File(out_path, 'w') as h5file:
        for file in glob.glob(os.path.join(data_path, '*_0.json')):
            print(file)
            id = os.path.split(file)[-1][:-len('_0.json')]
            with open(file.replace('.json','.jpg'),'rb') as fp:
                try:
                    img = Image.open(fp).convert("RGB")
                except:
                    print('error')
                    continue

            with open(file) as fp:
                js = json.load(fp)
            obj = js['objects']
            if len(obj) == 0:
                continue
            obj += add_trash_box(img, obj, localizer)

            faces = []
            sizes = []
            skus = []
            centers = []
            for o in obj:
                if 'rect' in o:
                    rect = np.array(o['rect'])
                else:
                    if 'polygon' in o:
                        rect = np.array(o['polygon'])
                    else:
                        continue
                if ('sku_id' in o) and (o['type'] == 'sku') and (o['sku_id'] != None) and o['sku_id'] not in skip_sku:
                    sku_id = o['sku_id']
                    if sku_id in sku_matching:
                        sku_id = sku_matching[sku_id]
                        print('replace')

                    y1, y2 = max(0, min(rect[:, 0])), max(0, min(img.size[1], max(rect[:, 0])))
                    x1, x2 = max(0, min(rect[:, 1])), max(0, min(img.size[0], max(rect[:, 1])))
                    w, h = x2 - x1, y2 - y1
                    if w >= 10 and h >= 10:
                        offset_x = w * k
                        offset_y = h * k
                        resize = int(size * (1 + 2 * k))
                        face = np.array(img.crop([x1 - offset_x, y1 - offset_y, x2 + offset_x, y2 + offset_y])
                                        .resize((resize, resize))).astype(np.uint8).T
                        faces.append(face)
                        skus.append(sku_id)
                        rect = np.array([[x1, y1], [x1 + w, y1], [x1 + w, y1 + h], [x1, y1 + h]], dtype=np.float32)
                        rect[:, 0] -= img.size[0] * 0.5
                        rect[:, 1] -= img.size[1] * 0.5
                        rect /= (img.size[0] + img.size[1]) * 0.25
                        centers.append(rect)
                        sizes.append([w, h])
            faces = np.array(faces)
            skus = np.array(skus)
            centers = np.array(centers)
            sizes = np.array(sizes)

            h5file.create_group(id)
            h5file[id].create_dataset('faces', data=faces)
            h5file[id].create_dataset('skus', data=skus.astype('S'))
            h5file[id].create_dataset('centers', data=centers)
            h5file[id].create_dataset('sizes', data=sizes)
            h5file[id].create_dataset('img_size', data=np.array(img.size))

localizer = load_localizer(os.path.join(workdir, 'params'))
parse_data(os.path.join('/Downloads', category, 'train'), os.path.join(workdir, category, out_folder, 'train.hdf5'), localizer)
parse_data(os.path.join('/Downloads', category, 'test'), os.path.join(workdir, category, out_folder, 'test.hdf5'), localizer)