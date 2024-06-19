import torch.nn
from torch.utils.data import TensorDataset, DataLoader
from torch.nn import Sequential, Linear, ReLU, MSELoss
from torch.optim import Adam
from utils import save_to_pickle
import numpy as np

seed = 123
np.random.seed(seed)


class AE(torch.nn.Module):
    def __init__(self, X_train, epochs, batch_size, lr, name):
        super(AE, self).__init__()
        self._X_train = torch.tensor(X_train, dtype=torch.float32)
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.name = name
        input_dim = X_train.shape[1]
        self.encoder = Sequential(
            Linear(in_features=input_dim, out_features=int(input_dim/2)),
            ReLU(),
            Linear(in_features=int(input_dim/2), out_features=int(input_dim/4)),
            ReLU(),
        )
        self.decoder = Sequential(
            Linear(in_features=int(input_dim/4), out_features=int(input_dim/2)),
            ReLU(),
            Linear(in_features=int(input_dim/2), out_features=input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        out = self.decoder(encoded)
        return out

    def train_autoencoder_content(self):
        """
        Method that builds and trains the autoencoder that processes the textual content
        """
        print("\nTraining autoencoder")
        loss_function = MSELoss()
        opt = Adam(self.parameters(), self.lr)
        ds = TensorDataset(self._X_train)
        dl = DataLoader(ds, batch_size=self.batch_size)
        best_loss = 999999999
        for epoch in range(self.epochs):
            self.train()
            total_loss = 0
            for batch in dl:
                out = self(batch[0])
                loss = loss_function(out, batch[0])
                opt.zero_grad()
                loss.backward()
                total_loss += loss
                opt.step()
            total_loss = total_loss/len(dl)
            if total_loss < best_loss:
                best_loss = total_loss
                print("Found best model at epoch {}. Loss: {}".format(epoch, best_loss))
                save_to_pickle(self.name, self)

    def predict(self, x):
        self.eval()
        return self(x)
