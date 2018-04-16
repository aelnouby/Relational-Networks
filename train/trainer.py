"""Trainer class for handling training flow
Author: Alaaeldin El-Nouby"""
import yaml
import torch
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.nn import DataParallel
from torchvision import transforms
from torch.autograd import Variable
from data.clevr_dataset import CLEVRDataset
from helpers.text_utils import TextHelper
from networks.relational_net import RelationalNetwork
# from networks.relational_net_3p import RelationNetworks as RelationalNetwork
from helpers.visualize import VisdomPlotter
from evaluation.metrics import MetricsInfo
from helpers.data_augmentation import RandomCrop, Rotation, HorizontalFlip
import argparse
with open('config.yaml') as f:
    config = yaml.load(f)


class Trainer(object):
    def __init__(self, lr, screen, batch_size):
        self.text_helper = TextHelper()
        self.rel_net = RelationalNetwork(vocab_size=self.text_helper.get_vocab_size()).cuda()
        self.rel_net = DataParallel(self.rel_net)
        augmenters = transforms.Compose([
            Rotation(),
            RandomCrop(),
            HorizontalFlip(),
        ])
        self.train_dataset = CLEVRDataset(os.path.join(config['clevr_hd5_path'], 'train.hdf5'), transforms=augmenters)
        self.val_dataset = CLEVRDataset(os.path.join(config['clevr_hd5_path'], 'val.hdf5'))
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=batch_size, num_workers=10, shuffle=True)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=64, num_workers=10, shuffle=False)
        self.optimizer = torch.optim.SGD(params=self.rel_net.parameters(), lr=lr)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.batch_size = batch_size

        self.num_epochs = 3000
        self.viz = VisdomPlotter(env_name=screen)
        self.metrics_tracker = MetricsInfo(env_name=screen)

    def train(self):
        iterations = 0
        for epoch in range(1, self.num_epochs):
            self.rel_net.train()

            for batch in tqdm(self.train_dataloader):
                images = batch['image']
                questions = batch['questions']
                answers = batch['answers']

                questions, questions_len, questions_sorting = self.text_helper.question_to_bow(questions)
                answers = self.text_helper.get_answer_index(answers)[questions_sorting]
                questions_sorting = torch.LongTensor(questions_sorting.copy())
                images = images[questions_sorting]

                images = Variable(images.cuda()).float()
                questions = Variable(torch.LongTensor(questions).cuda())
                answers = Variable(torch.LongTensor(answers).cuda())
                questions_len = Variable(torch.LongTensor(questions_len)).cuda()

                prediction = self.rel_net(images, questions, questions_len)

                loss = self.criterion(prediction, answers)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                self.metrics_tracker.track_accuracy(prediction, answers)
                iterations += 1

                if iterations % 10 == 0:
                    self.viz.plot('Loss', 'Train', iterations, loss.data.cpu().numpy()[0])
                    self.metrics_tracker.plot_accuracies(iterations)
                    self.viz.draw('sampeles', images[:8].data.cpu().numpy())

            self._validate(iterations)

    def _validate(self, epoch):
        self.rel_net.eval()
        for batch in tqdm(self.val_dataloader):
            images = batch['image']
            questions = batch['questions']
            answers = batch['answers']

            questions, questions_len, questions_sorting = self.text_helper.question_to_bow(questions)
            answers = self.text_helper.get_answer_index(answers)[questions_sorting]
            questions_sorting = torch.LongTensor(questions_sorting.copy())
            images = images[questions_sorting]

            images = Variable(images.cuda(), requires_grad=False).float()
            questions = Variable(torch.LongTensor(questions).cuda(), requires_grad=False)
            answers = Variable(torch.LongTensor(answers).cuda(), requires_grad=False)
            questions_len = Variable(torch.LongTensor(questions_len), requires_grad=False).cuda()

            prediction = self.rel_net(images, questions, questions_len)

            self.metrics_tracker.track_accuracy(prediction, answers)

        self.metrics_tracker.plot_accuracies(epoch, train=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=2.5e-4)
    parser.add_argument('--vis_screen', default='Relnet')
    args = parser.parse_args()
    trainer = Trainer(lr=args.lr, screen=args.screen)
    trainer.train()
