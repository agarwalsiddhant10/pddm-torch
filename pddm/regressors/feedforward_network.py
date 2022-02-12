# Copyright 2019 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#      http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import numpy as np


class FeedforwardNetwork(nn.Module):
    def __init__(self, inputSize, acSize, outputSize, num_fc_layers, depth_fc_layers):
        super(FeedforwardNetwork, self).__init__()
        self.inputSize = inputSize
        self.acSize = acSize
        self.outputSize = outputSize
        self.num_fc_layers = num_fc_layers
        self.depth_fc_layers = depth_fc_layers

        self.intermediate_size = depth_fc_layers
        self.model = []
        self.model.append(nn.Flatten())
        for i in range(num_fc_layers):
            if i == 0:
                self.model.append(nn.Linear(inputSize, self.intermediate_size))
                self.model.append(nn.ReLU())
            else:
                self.model.append(nn.Linear(self.intermediate_size, self.intermediate_size))
                self.model.append(nn.ReLU())

        self.model.append(nn.Linear(self.intermediate_size, outputSize))

        self.model =  nn.Sequential(*self.model)

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(m.weight, gain=torch.nn.init.calculate_gain('relu'))
                m.bias.data.fill_(0.0)

        self.model.apply(init_weights)

    
    def forward(self, inputs):
        # print(inputs.shape)
        first, second = torch.split(inputs, [(self.inputSize - self.acSize), self.acSize], 2)
        second = torch.clip(second, -1, 1)
        inputs = torch.cat((first, second), 2)

        return self.model(inputs)


class Ensemble(nn.Module):
    def __init__(self, ensemble_size, inputSize, acSize, outputSize, num_fc_layers, depth_fc_layers):
        super(Ensemble, self).__init__()
        self.ensemble_size = ensemble_size
        self.inputSize = inputSize
        self.acSize = acSize
        self.outputSize = outputSize
        self.num_fc_layers = num_fc_layers
        self.depth_fc_layers = depth_fc_layers

        self.intermediate_size = depth_fc_layers
        self.criterion = nn.MSELoss()

        self.members = []
        for i in range(self.ensemble_size):
            self.members.append(FeedforwardNetwork(inputSize, acSize, outputSize, num_fc_layers, depth_fc_layers))

        self.members = nn.ModuleList(self.members)

    def forward_train(self, inputs, labels): #[bs, K, sa]
        inputs_tiled = torch.tile(inputs, (self.ensemble_size, 1, 1, 1))

        loss = 0
        for i in range(self.ensemble_size):
            model = self.members[i].train()
            output_i = model(inputs_tiled[i])
            loss += self.criterion(output_i, labels)
    
        return loss/self.ensemble_size

    def forward_eval(self, inputs, labels): #[bs, K, sa]
        inputs_tiled = torch.tile(inputs, (self.ensemble_size, 1, 1, 1))

        with torch.no_grad():
            loss = 0
            for i in range(self.ensemble_size):
                model = self.members[i].eval()
                output_i = model(inputs_tiled[i])
                loss += self.criterion(output_i, labels)
    
        return loss/self.ensemble_size 

    def forward_sim(self, inputs): #[ens, bs, K, sa]
        with torch.no_grad():
            outputs = []
            for i in range(self.ensemble_size):
                model = self.members[i].eval()
                output_i = model(inputs[i])
                outputs.append(output_i.cpu().detach().numpy())
    
        return np.array(outputs)