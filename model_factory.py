################################################################################
# CSE 151B: Programming Assignment 3
# Code snippet by Ajit Kumar, Savyasachi
# Updated by Rohin, Yash, James
# Fall 2022
################################################################################
import torch
from torchvision.models import resnet50
import torch.nn as nn
import torch.nn.functional as F

import torch
from torchvision.models import resnet50
import torch.nn as nn
import torch.nn.functional as F


class CustomCNN(nn.Module):
    '''
    A Custom CNN (Task 1) implemented using PyTorch modules based on the architecture in the PA writeup.
    This will serve as the encoder for our Image Captioning problem.
    '''

    def __init__(self, outputs):
        '''
        Define the layers (convolutional, batchnorm, maxpool, fully connected, etc.)
        with the correct arguments

        Parameters:
            outputs => the number of output classes that the final fully connected layer
                       should map its input to
        '''
        num_classes = outputs

        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=11, stride=4)  # +BN+ReLU
        self.conv1_bn = nn.BatchNorm2d(64)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding=2)  # +BN+ReLU
        self.conv2_bn = nn.BatchNorm2d(128)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)  # +BN+ReLU
        self.conv3_bn = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)  # +BN+ReLU
        self.conv4_bn = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1)  # +BN+ReLU
        self.conv5_bn = nn.BatchNorm2d(128)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.adaptive_avgpool = nn.AdaptiveAvgPool2d(
            output_size=(1, 1))  # w/ kernel size of 1x1 it's not doing anything

        self.fc1 = nn.Linear(in_features=128, out_features=1024)  # +ReLU
        self.fc2 = nn.Linear(in_features=1024, out_features=1024)  # +ReLU
        self.fc3 = nn.Linear(in_features=1024, out_features=num_classes)
        # TODO

    def forward(self, x):
        '''
        Pass the input through each layer defined in the __init__() function
        in order.

        Parameters:
            x => Input to the CNN
        '''
        x = self.conv1(x)
        x = self.conv1_bn(x)
        x = F.relu(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.conv2_bn(x)
        x = F.relu(x)
        x = self.maxpool2(x)

        x = self.conv3(x)
        x = self.conv3_bn(x)
        x = F.relu(x)

        x = self.conv4(x)
        x = self.conv4_bn(x)
        x = F.relu(x)

        x = self.conv5(x)
        x = self.conv5_bn(x)
        x = F.relu(x)
        x = self.maxpool3(x)

        x = self.adaptive_avgpool(x)

        x = x.flatten(start_dim=1)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x


class CNN_LSTM(nn.Module):
    '''
    An encoder decoder architecture.
    Contains a reference to the CNN encoder based on model_type config value.
    Contains an LSTM implemented using PyTorch modules. This will serve as the decoder for our Image Captioning problem.
    '''
    def __init__(self, config_data, vocab):
        '''
        Initialize the embedding layer, LSTM, and anything else you might need.
        '''
        super(CNN_LSTM, self).__init__()
        self.vocab = vocab
        self.hidden_size = config_data['model']['hidden_size']
        self.embedding_size = config_data['model']['embedding_size']
        self.model_type = config_data['model']['model_type']
        self.max_length = config_data['generation']['max_length']
        self.deterministic = config_data['generation']['deterministic']
        self.temp = config_data['generation']['temperature']
        self.num_layers = 2

        # TODO
        self.embedding = nn.Embedding(num_embeddings=14463 ,embedding_dim=self.embedding_size)
        self.lstm = nn.LSTM(input_size=self.embedding_size,hidden_size=self.hidden_size,num_layers=self.num_layers,batch_first=True)
        # x -> (batch, seq, feature) from batch_first
        self.fc = nn.Linear(in_features=self.hidden_size,out_features=14463)


    def forward(self, images, captions, teacher_forcing=False):
        '''
        input:
            images: tensor of shape (Btach_size, embedding_size)
            captions: tensor of shape (Batch_size, Seq_len)
        Forward function for this model.
        If teacher forcing is true:
            - Pass encoded images to the LSTM at first time step.
            - Pass each encoded caption one word at a time to the LSTM at every time step after the first one.
        Else:
            - Pass encoded images to the LSTM at first time step.
            - Pass output from previous time step through the LSTM at subsequent time steps
            - Generate predicted caption from the output based on whether we are generating them deterministically or not.
        '''
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        hidden = torch.zeros(self.num_layers,images.size(0),self.hidden_size).to(device)
        cell = torch.zeros(self.num_layers,images.size(0),self.hidden_size).to(device)

        # captions = self.embedding(F.one_hot(captions,len(self.vocab.idx2word)))
        captions = self.embedding(captions)
        # One hot encode caption to shape: (batch,seq_len,num_classes)
        # Then embed to shape: (batch,seq,embedding.size)

        # captions = self.embedding(F.one_hot(captions,len(self.vocab.idx2word)).type(torch.float32))

        if teacher_forcing:
            # input = torch.cat((torch.reshape(images,(-1,1,self.embedding_size)),captions),dim=1)
            input = torch.cat((images.view(-1,1,self.embedding_size),captions),dim=1)
            out , (hidden,cell) = self.lstm(input,(hidden,cell))
            out = self.fc(out)
            return out
