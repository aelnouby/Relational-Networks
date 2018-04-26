"""Helper functions for text processing
Author: Alaaeldin El-Nouby"""
import nltk
import h5py
import yaml
import json
import numpy as np

with open('config.yaml') as f:
    config = yaml.load(f)


class TextHelper(object):
    def __init__(self):
        vocab_file = h5py.File(config['clevr_vocab'], 'r')
        self.questions_vocab = json.loads(str(np.array(vocab_file['questions_vocab'])))
        self.answers_vocab = json.loads(str(np.array(vocab_file['answers_vocab'])))
        self.inv_answers = {v: k for k, v in self.answers_vocab.items()}
        self.inv_questions = {v: k for k, v in self.questions_vocab.items()}

    def question_to_bow(self, questions):
        """
        Converts text question to list of indices representing each word index in the vocab.
        Each list must have the same size, so we pad zeroes until we reach the size of the
        longest question of the current batch
        :param questions: batch of Text questions
        :return: bow: batch of question word indices
        :return: questions_len: list of questions lengths
        :returnL questions_sorting: indices of sorted questions
        """
        questions_len = np.array([len(nltk.word_tokenize(x)) for x in questions])
        questions_sorting = np.argsort(questions_len)[::-1]
        max_len = questions_len[questions_sorting[0]]
        # Sort questions according to length, this step is important when calling 'pack_padded_sequence' function
        # before passing to the LSTM in networks/relational_net.py:forward()
        questions_sorted = np.array(questions)[questions_sorting]
        bows = np.zeros((len(questions), max_len))
        for idx, question in enumerate(questions_sorted):
            tokens = nltk.word_tokenize(question)
            indices = np.array([self.questions_vocab[t] for t in tokens])
            bows[idx, :len(tokens)] = indices

        questions_len = questions_len[questions_sorting]

        return bows.astype(int), questions_len, questions_sorting

    def bow_to_question(self, bows):
        """
        Converts a a list of indices from the vocab to a textual sentence
        :param bows: List of list of indices
        :return: questions: list of text-based questions
        """
        self.inv_questions[0] = ''
        questions = []
        for b in bows:
            question = [self.inv_questions[i] for i in b]
            question = ' '.join(question)
            questions.append(question)

        return questions

    def get_answer_index(self, answers):
        """
        Gets the answer's index from the Answers vocab
        :param answers: batch of text answers
        :return: batch of corresponding indices in answers vocb
        """
        indices = [self.answers_vocab[a] for a in answers]
        return np.array(indices)

    def get_answer(self, indicies):
        """
        Gets the answer's index from the Answers vocab
        :param indicies: list of indices (answers keys in vocab)
        :return: array of textual answers
        """
        answers = [self.inv_answers[i] for i in indicies]
        return np.array(answers)

    def get_vocab_size(self):
        """
        :return: Vocabulary size
        """
        return len(self.questions_vocab)
