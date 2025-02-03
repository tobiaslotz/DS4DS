import h5py
import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
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
NUM_EPOCHS = 15
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

# merge all datasets
X1X2X3_train = torch.vstack([X1[:TRAIN_INDEX, :, :], X2[:TRAIN_INDEX, :, :], X3[:TRAIN_INDEX, :, :]])
X1X2X3_test = torch.vstack([X1[TRAIN_INDEX:, :, :], X2[TRAIN_INDEX:, :, :], X3[TRAIN_INDEX:, :, :]])
y1y2y3_train = torch.vstack([y1[:TRAIN_INDEX, :], y2[:TRAIN_INDEX, :], y3[:TRAIN_INDEX, :]])
y1y2y3_test = torch.vstack([y1[TRAIN_INDEX:, :], y2[TRAIN_INDEX:, :], y3[TRAIN_INDEX:, :]])

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
    
merged_train_set = Trajectories(X1X2X3_train, y1y2y3_train)
merged_test_set = Trajectories(X1X2X3_test, y1y2y3_test)

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
X1X2X3_train_loader = DataLoader(merged_train_set, batch_size=BATCH_SIZE, shuffle=True)
X1X2X3_test_loader = DataLoader(merged_test_set, batch_size=BATCH_SIZE, shuffle=False)

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
        out = F.tanh(self.fc(out[:,-1,:]))
        return out
    
def laplacian_2d(x, grid_size):
    """
    Approximation des Laplace-Operators mit einer 5-Punkt-Diskretisierung.
    """
    x = x.view(-1, grid_size, grid_size)
    laplace_x = (
        -4 * x +
        torch.roll(x, shifts=1, dims=1) + torch.roll(x, shifts=-1, dims=1) +
        torch.roll(x, shifts=1, dims=2) + torch.roll(x, shifts=-1, dims=2)
        ) / (grid_size**2)
    return laplace_x.view(-1, grid_size * grid_size)

def cahn_hilliard(x, grid_size, D=3, gamma=0.5):
    laplace_x = laplacian_2d(x, grid_size)
    laplace_x3 = laplacian_2d(x**3, grid_size)
    laplace_laplace_x = laplacian_2d(laplace_x, grid_size)
    return D * (laplace_x3 - laplace_x - gamma * laplace_laplace_x)

def pinn_loss(y_true, y_pred, grid_size=100):
    mse_loss = F.mse_loss(y_pred, y_true)
    ch_loss = F.mse_loss(cahn_hilliard(y_pred, grid_size), cahn_hilliard(y_true, grid_size))
    return mse_loss + ch_loss

def train_model(train_loader, test_loader, config, dataset_name, verbose=True, use_pinn=False, num_epochs=NUM_EPOCHS):
    model = PINN(config['hidden_size'], config['num_layers'])
    model.to(device)
    best_test_loss = 10_000

    loss_func = nn.MSELoss() # + PDE informed loss
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

    for epoch in range(num_epochs):
        model.train(True)
        if verbose:
            print(f"Starting epoch: {epoch + 1}")
            print(f"*******************************")
        running_loss = 0.0
        epoch_loss = 0.0

        for i, batch in enumerate(train_loader):
            if use_pinn:
                X_batch, y_batch = torch.autograd.Variable(batch[0].to(device), requires_grad=True), batch[1].to(device)
            else:
                X_batch, y_batch = batch[0].to(device), batch[1].to(device)
            y_pred = model(X_batch)
            loss = loss_func(y_pred, y_batch)
            optimizer.zero_grad()
            if use_pinn:
                loss.backward(retain_graph=True)
                dxdt = X_batch.grad[:,-1,:]
                loss += ((dxdt - cahn_hilliard(y_pred, 100))**2).mean()
                optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss
            epoch_loss += loss

            if i % 10 == 0 and i > 0 and verbose:
                avg_loss = running_loss / 10
                print(f"Batch {i}, Loss: {avg_loss : .3f}")
                running_loss = 0.0

        if verbose:
            print(f"*******************************")
        
        val_loss = eval_model(test_loader, model, loss_func, verbose=verbose)
        if val_loss < best_test_loss:
            best_test_loss = val_loss
            if use_pinn:
                torch.save(model, f"PINN_{dataset_name}")
            else:
                torch.save(model, f"LSTM_{dataset_name}")

    return best_test_loss


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


def main(use_x1=False, use_x2=False, use_x3=False, use_all=False, use_pinn=False, num_epochs=NUM_EPOCHS):
    if not use_x1 and not use_x2 and not use_x3 and not use_all:
        return
    best_loss_x1 = best_loss_x2 = best_loss_x3 = best_loss_all = 10_000
    best_config_x1 = best_config_x2 = best_config_x3 = best_config_all = {}
    #param_space = list(product([5e-3, 1e-3, 1e-2], [5096, 4096, 2048, 1024], [1, 2]))
    param_space = [(1e-3, 2048, 1)]
    for lr, hs, nl in tqdm(param_space):
        param_config = {"lr": lr, "hidden_size" : hs, "num_layers" : nl}
        
        if use_x1:
            loss_x1 = train_model(X1_train_loader, X1_test_loader, 
                                  param_config, "X1", verbose=True, use_pinn=use_pinn, num_epochs=num_epochs)
        if use_x2:
            loss_x2 = train_model(X2_train_loader, X2_test_loader, 
                                  param_config, "X2", verbose=True, use_pinn=use_pinn, num_epochs=num_epochs)
        if use_x3:
            loss_x3 = train_model(X3_train_loader, X3_test_loader, 
                                  param_config, "X3", verbose=True, use_pinn=use_pinn, num_epochs=num_epochs)
        if use_all:
            loss_all = train_model(X1X2X3_train_loader, X1X2X3_test_loader, 
                                   param_config, "X1X2X3", verbose=True, use_pinn=use_pinn, num_epochs=num_epochs)
        
        if use_x1 and loss_x1 < best_loss_x1:
            best_loss_x1 = loss_x1
            best_config_x1 = param_config
        if use_x2 and loss_x2 < best_loss_x2:
            best_loss_x2 = loss_x2
            best_config_x2 = param_config
        if use_x3 and loss_x3 < best_loss_x3:
            best_loss_x3 = loss_x3
            best_config_x3 = param_config
        if use_all and loss_all < best_loss_all:
            best_loss_all = loss_all
            best_config_all = param_config

    if use_x1:
        print(f"X1 ==> best config: {best_config_x1}   test loss ==> {best_loss_x1 : .3f}")

    if use_x2:
        print(f"X2 ==> best config: {best_config_x2}   test loss ==> {best_loss_x2 : .3f}")

    if use_x3:
        print(f"X3 ==> best config: {best_config_x3}   test loss ==> {best_loss_x3 : .3f}")

    if use_all:
        print(f"X1X2X3 ==> best config: {best_config_all}   test loss ==> {best_loss_all : .3f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--X1", action="store_true")
    parser.add_argument("--X2", action="store_true")
    parser.add_argument("--X3", action="store_true")
    parser.add_argument("--X1X2X3", action="store_true")
    parser.add_argument("--pinn", action="store_true")
    parser.add_argument("--n_epochs", type=int, default=15)
    args = parser.parse_args()
    main(args.X1, args.X2, args.X3, args.X1X2X3, args.pinn, args.n_epochs)
