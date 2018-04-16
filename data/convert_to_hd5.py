"""Script to convert dataset to HD5 format
Author: Alaaeldin El-Nouby
"""
import argparse
import yaml
import h5py
import os
import json
from joblib import Parallel, delayed
import multiprocessing
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
        for entry in tqdm(question_info):
            pair = tuple((entry['question'], entry['answer']))
            if entry['image_filename'] not in self.image_question_mapping:
                self.image_question_mapping[entry['image_filename']] = []
                self.dataset.create_group(str(entry['image_index']))

            self.image_question_mapping[entry['image_filename']].append(pair)

        num_cores = multiprocessing.cpu_count() - 1
        print('Multi-processing writing...')
        Parallel(n_jobs=1)(delayed(self._write_example)(it, key)
                           for it, key in tqdm(enumerate(self.image_question_mapping.keys()), total=len(self.image_question_mapping)))

    def _write_example(self, it, key):
        img = open(os.path.join(self.images_path, key), 'rb').read()
        img = cv2.imdecode(np.fromstring(img, np.uint8), 1)
        img = cv2.resize(img, (128, 128))
        _, img = cv2.imencode('.jpg', img)
        img = img.tostring()

        questions = [x[0] for x in self.image_question_mapping[key]]
        answers = [x[1] for x in self.image_question_mapping[key]]

        example = self.dataset[str(it)]
        example.create_dataset('image', data=[img])
        example.create_dataset('questions', data=json.dumps(questions))
        example.create_dataset('answers', data=json.dumps(answers))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', default='train')

    args = parser.parse_args()
    converter = HD5Converter(split=args.split)
    converter.convert_to_hd5()
