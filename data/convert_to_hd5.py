"""Script to convert dataset to HD5 format
Author: Alaaeldin El-Nouby
"""
import argparse
import yaml
import h5py
import os
import json
from tqdm import tqdm
import cv2
import numpy as np

with open('config.yaml') as f:
    config = yaml.load(f)


class HD5Converter(object):
    def __init__(self, split='train'):
        self.images_path = os.path.join(config['clevr_images_path'], split)
        self.question_path = os.path.join(config['clevr_questions_path'], 'CLEVR_{}_questions.json'.format(split))

        hd5_path = os.path.join(config['clevr_hd5_path'], '{}.hdf5'.format(split))
        self.dataset = h5py.File(hd5_path, 'w')
        self.image_question_mapping = {}

    def convert_to_hd5(self):
        question_info = json.load(open(self.question_path))['questions']
        print('Creating mapping...')
        i = 0
        for entry in tqdm(question_info):
            self._write_example(i, entry['image_filename'], entry['question'], entry['answer'])
            i += 1

    def _write_example(self, it, image, question, answer):

        img = open(os.path.join(self.images_path, image), 'rb').read()
        img = cv2.imdecode(np.fromstring(img, np.uint8), 1)
        img = cv2.resize(img, (128, 128))
        _, img = cv2.imencode('.jpg', img)
        img = img.tostring()

        example = self.dataset.create_group(str(it))
        example.create_dataset('image', data=[img])
        example.create_dataset('question', data=json.dumps(question))
        example.create_dataset('answer', data=json.dumps(answer))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', default='val')

    args = parser.parse_args()
    converter = HD5Converter(split=args.split)
    converter.convert_to_hd5()
