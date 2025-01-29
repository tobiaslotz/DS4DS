import h5py
import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as f
import argparse
#import matplotlib.pyplot as plt
#from ray import tune
from torch.utils.data import Dataset, DataLoader
from itertools import product
from tqdm import tqdm

# DATA PREPING
data = h5py.File("data/CH_D3.jld2", "r")
TRAIN_INDEX = 260
BATCH_SIZE = 5
NUM_EPOCHS = 10
X1_CH = torch.Tensor(np.array(data["X1"]))
X2_CH = torch.Tensor(np.array(data["X2"]))
X3_CH = torch.Tensor(np.array(data["X3"]))

dt_CH = 0.1

y1 = X1_CH[101:, :]
y2 = X2_CH[101:, :]
y3 = X3_CH[101:, :]
X1 = torch.stack([X1_CH[i:i+100, :] for i in range(1, 301)])
X2 = torch.stack([X2_CH[i:i+100, :] for i in range(1, 301)])
X3 = torch.stack([X3_CH[i:i+100, :] for i in range(1, 301)])

# Shapes:
# X: (300, 100, 10_000)
# y: (300, 10_000)

class Trajectories(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, i):
        return self.X[i], self.y[i]

def create_train_test_loader(X, y, train_index=TRAIN_INDEX, batch_size=BATCH_SIZE):
    y_train = y[:train_index, :]
    y_test = y[train_index:, :]
    X_train = X[:train_index, :, :]
    X_test = X[train_index:, :, :]

    train_set = Trajectories(X_train, y_train)
    test_set = Trajectories(X_test, y_test)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

X1_train_loader, X1_test_loader = create_train_test_loader(X1, y1)
X2_train_loader, X2_test_loader = create_train_test_loader(X2, y2)
X3_train_loader, X3_test_loader = create_train_test_loader(X3, y3)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

INPUT_SIZE = OUTPUT_SIZE = 10_000
seq_length = 100

class PINN(nn.Module):
    def __init__(self, hidden_size, num_layers, batch_size=BATCH_SIZE):
        super().__init__()
        # input shape: (Batch, Seq_length, input_size)
        self.lstm = nn.LSTM(INPUT_SIZE, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, OUTPUT_SIZE)
        self.h0 = torch.zeros(num_layers, batch_size, hidden_size).to(device)
        self.c0 = torch.zeros(num_layers, batch_size, hidden_size).to(device)

    def forward(self, x):
        out, _ = self.lstm(x, (self.h0, self.c0))
        out = f.tanh(self.fc(out[:,-1,:]))
        return out


def train_model(train_loader, test_loader, config, verbose=True):
    model = PINN(config['hidden_size'], config['num_layers'])
    model.to(device)
    best_state_dict = {}
    best_test_loss = 10_000

    loss_func = nn.MSELoss() # + PDE informed loss
    #optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'])
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

    for epoch in range(NUM_EPOCHS):
        model.train(True)
        #if verbose:
        print(f"Starting epoch: {epoch + 1}")
        print(f"*******************************")
        running_loss = 0.0
        epoch_loss = 0.0

        for i, batch in enumerate(train_loader):
            X_batch, y_batch = batch[0].to(device), batch[1].to(device)
            y_pred = model(X_batch)
            loss = loss_func(y_pred, y_batch)
            running_loss += loss
            epoch_loss += loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 10 == 0 and i > 0 and verbose:
                avg_loss = running_loss / 10
                print(f"Batch {i}, Loss: {avg_loss : .3f}")
                running_loss = 0.0

        #if verbose:
        print(f"*******************************")
        
        val_loss = eval_model(test_loader, model, loss_func, verbose=verbose)
        if val_loss < best_test_loss:
            best_test_loss = val_loss
            best_state_dict = model.state_dict()
    
    return PINN(config['hidden_size'], config['num_layers']).load_state_dict(best_state_dict), best_test_loss


def eval_model(test_loader, model, loss_func, verbose=True):
    model.train(False)
    running_loss = 0.0

    for i, batch in enumerate(test_loader):
        X_batch, y_batch = batch[0].to(device), batch[1].to(device)

        with torch.no_grad():
            y_pred = model(X_batch)
            loss = loss_func(y_pred, y_batch)
            running_loss += loss

    avg_loss = running_loss / (i+1)

    if verbose:
        print(f"Val loss: {avg_loss: .3f}")
    
    return avg_loss


def main(use_x1=False, use_x2=False, use_x3=False):
    if not use_x1 and not use_x2 and not use_x3:
        return
    best_loss_x1 = best_loss_x2 = best_loss_x3 = 10_000
    best_config_x1 = best_config_x2 = best_config_x3 = {}
    best_model_x1 = best_model_x2 = best_model_x3 = None
    param_space = list(product([5e-3, 1e-3, 1e-2], [5096, 4096, 2048, 1024], [1, 2]))
    #param_space = [(1e-3, 5096, 1)]
    #param_space = [(1e-3, 4048, 1)]
    for lr, hs, nl in tqdm(param_space):
        param_config = {"lr": lr, "hidden_size" : hs, "num_layers" : nl}
        
        if use_x1:
            model_x1, loss_x1 = train_model(X1_train_loader, X1_test_loader, param_config, verbose=True)
        if use_x2:
            model_x2, loss_x2 = train_model(X2_train_loader, X2_test_loader, param_config, verbose=True)
        if use_x3:
            model_x3, loss_x3 = train_model(X3_train_loader, X3_test_loader, param_config, verbose=True)
        
        if use_x1 and loss_x1 < best_loss_x1:
            best_loss_x1 = loss_x1
            best_model_x1 = model_x1
            best_config_x1 = param_config
        if use_x2 and loss_x2 < best_loss_x2:
            best_loss_x2 = loss_x2
            best_model_x2 = model_x2
            best_config_x2 = param_config
        if use_x3 and loss_x3 < best_loss_x3:
            best_loss_x3 = loss_x3
            best_model_x3 = model_x3
            best_config_x3 = param_config

    if use_x1:
        print(f"X1 ==> best config: {best_config_x1}")
        torch.save(best_model_x1, "LSTM_X1")

    if use_x2:
        print(f"X2 ==> best config: {best_config_x2}")
        torch.save(best_model_x2, "LSTM_X2")

    if use_x3:
        print(f"X3 ==> best config: {best_config_x3}")
        torch.save(best_model_x3, "LSTM_X3")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--X1", action="store_true")
    parser.add_argument("--X2", action="store_true")
    parser.add_argument("--X3", action="store_true")
    args = parser.parse_args()
    main(args.X1, args.X2, args.X3)
