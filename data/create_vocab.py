import h5py
import nltk
from tqdm import tqdm
import json
import yaml
import os

with open('config.yaml') as f:
    config = yaml.load(f)

data = h5py.File(os.path.join(config['clevr_hd5_path'], 'train.hdf5'), 'a')
question_path = os.path.join(config['clevr_questions_path'], 'CLEVR_{}_questions.json'.format('train'))
questions = json.load(open(question_path))['questions']

word_dic = {}
answer_dic = {}
# Starting from one because we pad with zeroes
word_index = 1
answer_index = 0

for question in tqdm(questions):
    words = nltk.word_tokenize(question['question'])

    for word in words:
        if word not in word_dic:
            word_dic[word] = word_index
            word_index += 1

    answer_word = question['answer']

    if answer_word not in answer_dic:
        answer_dic[answer_word] = answer_index
        answer_index += 1


vocab = h5py.File(config['clevr_vocab'], 'w')
vocab.create_dataset('questions_vocab', data=json.dumps(word_dic))
vocab.create_dataset('answers_vocab', data=json.dumps(answer_dic))
