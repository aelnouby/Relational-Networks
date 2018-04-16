"""Relational Network Implementation
Author: Alaaeldin El-Nouby"""

import torch.nn as nn
import torch


class RelationalNetwork(nn.Module):
    def __init__(self, vocab_size, conv_kernels=24, lstm_hidden=128, embedding_dim=32, g_dim=256, f_dim=256, num_classes=28):
        super(RelationalNetwork, self).__init__()
        self.lstm_hidden = lstm_hidden
        self.g_dim = g_dim

        self.object_encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=conv_kernels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(conv_kernels),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=conv_kernels, out_channels=conv_kernels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(conv_kernels),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=conv_kernels, out_channels=conv_kernels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(conv_kernels),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=conv_kernels, out_channels=conv_kernels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(conv_kernels),
            nn.LeakyReLU(negative_slope=0.1)
        )

        self.embedding_layer = nn.Embedding(num_embeddings=vocab_size+1, embedding_dim=embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=lstm_hidden, batch_first=True)

        self.g = nn.Sequential(
            nn.Linear(conv_kernels*2 + lstm_hidden, g_dim),
            nn.ReLU(),
            nn.Linear(g_dim, g_dim),
            nn.ReLU(),
            nn.Linear(g_dim, g_dim),
            nn.ReLU(),
            nn.Linear(g_dim, g_dim),
            nn.ReLU()
        )

        self.f = nn.Sequential(
            nn.Linear(g_dim, f_dim),
            nn.ReLU(),
            nn.Linear(f_dim, f_dim),
            nn.Dropout(p=0.02),
            nn.ReLU(),
            nn.Linear(f_dim, num_classes)
        )

    # Code follows https://github.com/rosinality/relation-networks-pytorch/blob/master/model.py to some extent
    def forward(self, image, question, questions_len):
        object_encoding = self.object_encoder(image)
        batch_size, num_channels, width, height = object_encoding.size()
        num_objects = width * height

        word_vecs = self.embedding_layer(question)
        questions_len = list(questions_len.data.cpu().numpy())
        word_vecs_packed = nn.utils.rnn.pack_padded_sequence(word_vecs, questions_len, batch_first=True)

        _, (h, _) = self.lstm(word_vecs_packed)

        object_encoding = object_encoding.permute(0, 2, 3, 1).contiguous()  # Channel Last
        object_encoding = object_encoding.view(batch_size, num_objects, num_channels)  # (batch, 64, 24)
        object_encoding = object_encoding.unsqueeze(2).repeat(1, 1, num_objects, 1)  # (batch, 64, 64, 24)

        objects_1 = object_encoding.view(batch_size, num_objects * num_objects, num_channels)
        objects_2 = object_encoding.permute(0, 2, 1, 3).contiguous().view(batch_size, num_objects * num_objects, num_channels)

        question_embedding = h.permute(1, 0, 2).contiguous().repeat(1, num_objects*num_objects, 1)

        relation_dim = num_channels * 2 + self.lstm_hidden
        relation_vector = torch.cat([objects_1, objects_2, question_embedding], dim=2).view(-1, relation_dim)

        relation = self.g(relation_vector)

        relation = relation.view(batch_size, num_objects * num_objects, self.g_dim)  # (batch, 64x64, 256)
        relation = relation.sum(dim=1)

        output = self.f(relation)

        return output
