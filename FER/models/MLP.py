import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, layers, output_dim, dropout=0.2, batch_norm=False, activation='relu'):
        super(MLP, self).__init__()
        self.in_fc = nn.Linear(input_dim, layers[0])
        # layers -> list of neuros in each layer this is used for hidden layers
        self.hidden_layers = nn.ModuleList([nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)])
        self.out_fc = nn.Linear(layers[-1], output_dim)
        self.dropout = nn.Dropout(dropout)
        self.batch_norm = batch_norm
        if self.batch_norm:
            self.bn_layers = nn.ModuleList([nn.BatchNorm1d(layer) for layer in layers])
            
        if activation == 'ReLU':
            self.act = nn.ReLU()
        elif activation == 'LeakyReLU':
            self.act = nn.LeakyReLU()
        else:
            raise NotImplementedError

    def forward(self, x):
        if self.batch_norm:
            x = self.dropout(self.act(self.bn_layers[0](self.in_fc(x))))
        else:
            x = self.dropout(self.act(self.in_fc(x)))
        
        for i, hidden_layer in enumerate(self.hidden_layers):
            if self.batch_norm:
                x = self.dropout(self.act(self.bn_layers[i+1](hidden_layer(x))))
            else:
                x = self.dropout(self.act(hidden_layer(x)))
        x = self.out_fc(x)
        return x
    
    def extract_feature_from_layer(self, x, layer_num):
        x = self.act(self.in_fc(x))
        for hidden_layer in self.hidden_layers[:layer_num]:
            x = self.act(hidden_layer(x))
        x = self.hidden_layers[layer_num](x)
        return x
    

