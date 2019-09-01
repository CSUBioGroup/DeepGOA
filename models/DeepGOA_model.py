#-*- encoding:utf8 -*-

import fire
import os
import time
import sys

import torch as t
from torch import nn
from torch.autograd import Variable

sys.path.append("../")
from models.BasicModule import BasicModule
from utils.config import DefaultConfig


configs = DefaultConfig()


class LSTMLayer(BasicModule):
    def __init__(self, ):
        super(LSTMLayer, self).__init__()
        input_size = configs.features_W
        self.output_size = 1
        batch_size = configs.batch_size

        dropout = configs.dropout
        self.lstm = nn.LSTM(input_size, self.output_size,
                            num_layers=2,
                            dropout=dropout,
                            batch_first=True, bidirectional=True)

    def forward(self, x):
        output, (_, _) = self.lstm(x)
        shapes = output.data.shape
        features = output.contiguous()
        features = features.view(shapes[0],shapes[1]*shapes[2])
        return features

class LSTMEncoder(BasicModule):
    def __init__(self,):
        super(LSTMEncoder,self).__init__()
        input_size = configs.features_W
        self.output_size = configs.lstm_hidden
        batch_size = configs.batch_size

        dropout = configs.dropout
        self.lstm = nn.LSTM(input_size,self.output_size,num_layers = 2,dropout=dropout,batch_first=True,bidirectional=True)
        
        
    def forward(self,x):
        output,(_,_) = self.lstm(x)
        features = t.unsqueeze(output,1)
        return features
        
        
class ConvsLayer(BasicModule):
    def __init__(self,):

        super(ConvsLayer,self).__init__()
        
        self.kernels = configs.kernels
        hidden_channels = configs.hidden_channels
        in_channel = 1
        features_W = configs.features_W
        features_L = configs.features_L
        W_size = configs.lstm_hidden * 2

        padding1 = (self.kernels[0]-1)//2
        padding2 = (self.kernels[1]-1)//2
        padding3 = (self.kernels[2]-1)//2
        self.conv1 = nn.Sequential()
        self.conv1.add_module("conv1",
            nn.Conv2d(in_channel, hidden_channels,
            padding=(padding1,0),
            kernel_size=(self.kernels[0],W_size)))
        self.conv1.add_module("ReLU",nn.PReLU())
        self.conv1.add_module("pooling1",nn.MaxPool2d(kernel_size=(features_L,1),stride=1))
        
        self.conv2 = nn.Sequential()
        self.conv2.add_module("conv2",
            nn.Conv2d(in_channel, hidden_channels,
            padding=(padding2,0),
            kernel_size=(self.kernels[1],W_size)))
        self.conv2.add_module("ReLU",nn.ReLU())
        self.conv2.add_module("pooling2",nn.MaxPool2d(kernel_size=(features_L,1),stride=1))
        
        self.conv3 = nn.Sequential()
        self.conv3.add_module("conv3",
            nn.Conv2d(in_channel, hidden_channels,
            padding=(padding3,0),
            kernel_size=(self.kernels[2],W_size)))
        self.conv3.add_module("ReLU",nn.ReLU())
        self.conv3.add_module("pooling3",nn.MaxPool2d(kernel_size=(features_L,1),stride=1))
    
    def forward(self,x):
        features1 = self.conv1(x)
        features2 = self.conv2(x)
        features3 = self.conv3(x)

        features = t.cat((features1,features2,features3),1)
        shapes = features.data.shape
        features = features.view(shapes[0],shapes[1]*shapes[2]*shapes[3])
        
        return features


class ConvsLayer_only(BasicModule):
    def __init__(self, ):
        super(ConvsLayer_only, self).__init__()

        self.kernels = configs.kernels
        hidden_channels = configs.hidden_channels
        in_channel = 1
        features_W = configs.features_W
        features_L = configs.features_L
        W_size = configs.lstm_hidden * 2

        padding1 = (self.kernels[0] - 1) // 2
        padding2 = (self.kernels[1] - 1) // 2
        padding3 = (self.kernels[2] - 1) // 2

        pool_size1 = features_L + 2 * padding1 - self.kernels[0] + 1
        pool_size2 = features_L + 2 * padding2 - self.kernels[1] + 1
        pool_size3 = features_L + 2 * padding3 - self.kernels[2] + 1

        self.conv1 = nn.Sequential()
        self.conv1.add_module("conv1",
                              nn.Conv2d(in_channel, hidden_channels,
                                        padding=(padding1, 0),
                                        kernel_size=(self.kernels[0], W_size)))
        self.conv1.add_module("ReLU", nn.PReLU())
        self.conv1.add_module("pooling1", nn.MaxPool2d(kernel_size=(pool_size1, 1), stride=1))

        self.conv2 = nn.Sequential()
        self.conv2.add_module("conv2",
                              nn.Conv2d(in_channel, hidden_channels,
                                        padding=(padding2, 0),
                                        kernel_size=(self.kernels[1], W_size)))
        self.conv2.add_module("ReLU", nn.ReLU())
        self.conv2.add_module("pooling2", nn.MaxPool2d(kernel_size=(pool_size2, 1), stride=1))

        self.conv3 = nn.Sequential()
        self.conv3.add_module("conv3",
                              nn.Conv2d(in_channel, hidden_channels,
                                        padding=(padding3, 0),
                                        kernel_size=(self.kernels[2], W_size)))
        self.conv3.add_module("ReLU", nn.ReLU())
        self.conv3.add_module("pooling3", nn.MaxPool2d(kernel_size=(pool_size3, 1), stride=1))

    def forward(self, x):
        features1 = self.conv1(x)
        features2 = self.conv2(x)
        features3 = self.conv3(x)

        features = t.cat((features1, features2, features3), 1)
        shapes = features.data.shape
        features = features.view(shapes[0], shapes[1] * shapes[2] * shapes[3])

        return features


class DeepGOA(BasicModule):
    def __init__(self, class_nums, interPro_size,tag="bp"):
        super(DeepGOA, self).__init__()
        global configs
        configs.hidden_channels = 400

        self.batch_size = configs.batch_size
        self.dropout = configs.dropout
        self.embeddings_num = configs.embedding_num
        self.embeddings_num = 256

        self.layers = nn.Sequential()
        self.layers.add_module("layer_lstm",
                               LSTMEncoder())
        self.layers.add_module("layer_convs",
                               ConvsLayer())

        self.DNN = nn.Sequential()
        self.DNN.add_module("DNN_layer1",
                            nn.Linear(35020, interPro_size[0]))
        self.DNN.add_module("Sigmoid1",
                            nn.Sigmoid())
        self.DNN.add_module("DNN_layer2",
                            nn.Linear(interPro_size[0], interPro_size[1]))
        if tag != "mf":
            self.DNN.add_module("Sigmoid2",
                            nn.Sigmoid())

        self.features_num = configs.hidden_channels * 3 + interPro_size[1]
        self.sequence_linear = nn.Sequential()
        self.sequence_linear.add_module("DNN_layer1",
                                        nn.Linear(self.features_num, 256))
        self.sequence_linear.add_module("ReLU",
                                        nn.ReLU())

        self.features_num = 256 + self.embeddings_num
        self.Linear = nn.Sequential(
            nn.Linear(self.features_num, 1024),
            nn.ReLU())
        self.dropout = nn.Dropout(self.dropout)

        self.outLayer = nn.Sequential(
            nn.Linear(1024, class_nums),
            nn.Sigmoid())

    def forward(self, x1, x2, x3):
        features = self.layers(x1)
        features_DNN = self.DNN(x3)
        features = t.cat((features_DNN, features), 1)
        features = self.sequence_linear(features)
        features = t.cat((features, x2), 1)
        features = self.Linear(features)
        features = self.dropout(features)
        features = self.outLayer(features)

        return features

class DeepGOA_InterPro(BasicModule):
    def __init__(self,class_nums):
        super(DeepGOA_InterPro,self).__init__()
        global configs
        self.batch_size = configs.batch_size
        self.dropout = configs.dropout

        self.DNN = nn.Sequential()
        self.DNN.add_module("DNN_layer1",
                            nn.Linear(35020,1024))
        self.DNN.add_module("sigmoid1",
                            nn.Sigmoid())
        self.DNN.add_module("DNN_layer2",
                            nn.Linear(1024,512))
        self.DNN.add_module("sigmoid2",
                            nn.Sigmoid())
        self.features_num = 512
        self.Linear = nn.Sequential(
            nn.Linear(self.features_num, 1024),
            nn.ReLU())
        self.dropout = nn.Dropout(self.dropout)

        self.outLayer = nn.Sequential(
            nn.Linear(1024, class_nums),
            nn.Sigmoid())

    def forward(self, x1, x2,x3):
        features = self.DNN(x3)
        features = self.Linear(features)
        features = self.dropout(features)
        features = self.outLayer(features)

        return features


class DeepGOA_InterPro_PPI(BasicModule):
    def __init__(self,class_nums):
        super(DeepGOA_InterPro_PPI,self).__init__()
        self.batch_size = configs.batch_size
        self.dropout = 0.1
        self.embeddings_num = configs.embedding_num
        self.embeddings_num = 256

        self.DNN = nn.Sequential()
        self.DNN.add_module("DNN_layer1",
                            nn.Linear(35020, 1024))
        self.DNN.add_module("sigmoid1",
                            nn.Sigmoid())
        self.DNN.add_module("DNN_layer2",
                            nn.Linear(1024, 512))
        self.DNN.add_module("sigmoid2",
                            nn.Sigmoid())

        self.features_num = + self.embeddings_num + 512


        self.Linear = nn.Sequential(
            nn.Linear(self.features_num, 1024),
            nn.ReLU())
        self.dropout = nn.Dropout(self.dropout)

        self.outLayer = nn.Sequential(
            nn.Linear(1024, class_nums),
            nn.Sigmoid())

    def forward(self, x1, x2,x3):
        features_DNN = self.DNN(x3)
        features = t.cat((features_DNN, x2), 1)
        features = self.Linear(features)
        features = self.dropout(features)
        features = self.outLayer(features)

        return features


class DeepGOA_PPI(BasicModule):
    def __init__(self,class_nums):
        super(DeepGOA_PPI,self).__init__()
        global configs
        self.batch_size = configs.batch_size
        self.dropout = configs.dropout

        self.features_num = 256
        self.Linear = nn.Sequential(
            nn.Linear(self.features_num, 1024),
            nn.ReLU())

        self.outLayer = nn.Sequential(
            nn.Linear(1024, class_nums),
            nn.Sigmoid())

    def forward(self, x1, x2,x3):
        features = self.Linear(x2)
        features = self.outLayer(features)

        return features



class DeepGOA_Seq(BasicModule):
    def __init__(self, class_nums):
        super(DeepGOA_Seq, self).__init__()

        global configs
        configs.hidden_channels = 400

        self.batch_size = configs.batch_size
        self.dropout = configs.dropout
        self.embeddings_num = configs.embedding_num

        self.layers = nn.Sequential()
        self.layers.add_module("layer_lstm",
                               LSTMEncoder())
        self.layers.add_module("layer_convs",
                               ConvsLayer())

        self.features_num = configs.hidden_channels * 3
        self.Linear = nn.Sequential(
            nn.Linear(self.features_num, 1024),
            nn.ReLU())
        self.dropout = nn.Dropout(self.dropout)

        self.outLayer = nn.Sequential(
            nn.Linear(1024, class_nums),
            nn.Sigmoid())

    def forward(self, x1, x2,x3):
        features = self.layers(x1)
        features = self.Linear(features)
        features = self.dropout(features)
        features = self.outLayer(features)

        return features


class DeepGOA_Seq_BiLSTM(BasicModule):
    def __init__(self, class_nums):
        super(DeepGOA_Seq_BiLSTM, self).__init__()

        global configs

        self.batch_size = configs.batch_size
        self.dropout = configs.dropout
        self.embeddings_num = configs.embedding_num

        self.layers = nn.Sequential()
        self.layers.add_module("layer_lstm",
                               LSTMLayer())

        self.features_num = configs.features_L * 2 
        self.Linear = nn.Sequential(
            nn.Linear(self.features_num, 1024),
            nn.ReLU())
        self.dropout = nn.Dropout(self.dropout)

        self.outLayer = nn.Sequential(
            nn.Linear(1024, class_nums),
            nn.Sigmoid())

    def forward(self, x1, x2,x3):
        features = self.layers(x1)
        features = self.Linear(features)
        features = self.dropout(features)
        features = self.outLayer(features)

        return features

class DeepGOA_Seq_PPI(BasicModule):
    def __init__(self, class_nums):
        super(DeepGOA_Seq_PPI, self).__init__()

        global configs
        configs.hidden_channels = 400

        self.batch_size = configs.batch_size
        self.dropout = configs.dropout = 0.2
        self.embeddings_num = configs.embedding_num

        self.layers = nn.Sequential()
        self.layers.add_module("layer_lstm",
                               LSTMEncoder())
        self.layers.add_module("layer_convs",
                               ConvsLayer())

        self.embeddings_num = 256
        self.features_num = configs.hidden_channels * 3 + self.embeddings_num
        self.Linear = nn.Sequential(
            nn.Linear(self.features_num, 1024),
            nn.ReLU())
        self.dropout = nn.Dropout(self.dropout)

        self.outLayer = nn.Sequential(
            nn.Linear(1024, class_nums),
            nn.Sigmoid())

    def forward(self, x1, x2,x3):
        features = self.layers(x1)
        features = t.cat((features, x2), 1)
        features = self.Linear(features)
        features = self.dropout(features)
        features = self.outLayer(features)

        return features



class DeepGOA_Seq_Multi_CNN(BasicModule):
    def __init__(self, class_nums,kernels):
        super(DeepGOA_Seq_Multi_CNN, self).__init__()

        global configs
        configs.kernels = kernels
        configs.hidden_channels = 400

        self.batch_size = configs.batch_size
        self.dropout = configs.dropout
        self.embeddings_num = configs.embedding_num

        self.layers = nn.Sequential()
        self.layers.add_module("layer_convs",
                               ConvsLayer_only())

        self.features_num = configs.hidden_channels * 3 #+ self.embeddings_num
        self.Linear = nn.Sequential(
            nn.Linear(self.features_num, 1024),
            nn.ReLU())
        self.dropout = nn.Dropout(self.dropout)

        self.outLayer = nn.Sequential(
            nn.Linear(1024, class_nums),
            nn.Sigmoid())

    def forward(self, x1, x2,x3):
        features =  t.unsqueeze(x1,1)
        features = self.layers(features)
        features = self.Linear(features)
        features = self.dropout(features)
        features = self.outLayer(features)

        return features


class DeepGOA_Seq_InterPro(BasicModule):
    def __init__(self,class_nums):
        super(DeepGOA_Seq_InterPro,self).__init__()
        global configs
        configs.hidden_channels = 400

        self.batch_size = configs.batch_size
        self.dropout = configs.dropout 
        self.embeddings_num = configs.embedding_num
        self.embeddings_num = 256

        self.layers = nn.Sequential()
        self.layers.add_module("layer_lstm",
                               LSTMEncoder())
        self.layers.add_module("layer_convs",
                               ConvsLayer())

        self.DNN = nn.Sequential()
        self.DNN.add_module("DNN_layer1",
                            nn.Linear(35020,1024))
        self.DNN.add_module("sigmoid1",
                            nn.Sigmoid())
        self.DNN.add_module("DNN_layer2",
                            nn.Linear(1024,512))
        self.DNN.add_module("sigmoid2",
                            nn.Sigmoid())

        self.features_num = configs.hidden_channels * 3 + 512
        self.sequence_linear = nn.Sequential()
        self.sequence_linear.add_module("DNN_layer1",
                            nn.Linear(self.features_num,256))
        self.sequence_linear.add_module("ReLU",
                            nn.ReLU())

        self.features_num = 256
        self.Linear = nn.Sequential(
            nn.Linear(self.features_num, 1024),
            nn.ReLU())
        self.dropout = nn.Dropout(self.dropout)

        self.outLayer = nn.Sequential(
            nn.Linear(1024, class_nums),
            nn.Sigmoid())

    def forward(self, x1, x2,x3):
        features = self.layers(x1)
        features_DNN = self.DNN(x3)
        features = t.cat((features_DNN, features), 1)
        features = self.sequence_linear(features)
        features = self.Linear(features)
        features = self.dropout(features)
        features = self.outLayer(features)

        return features

