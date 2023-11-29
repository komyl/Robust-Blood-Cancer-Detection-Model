import torch
import torch.nn as nn
from weakly_supervised_localization import WSLoss

class Interpreter:
    def __init__(self, cnn, cam, loader):
        self.cnn = cnn
        self.cam = cam
        self.loader = loader
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.ws_loss_fn = WSLoss(self.loss_fn)
        self.optimizer = torch.optim.Adam(self.cnn.parameters())

    def train_weakly_supervised(self):
        for images, labels in self.loader:
            self.optimizer.zero_grad()

            preds = self.cnn(images)
            cam_maps = self.cam(images)

            loss = self.ws_loss_fn(preds, cam_maps, labels)
            loss.backward()

            self.optimizer.step()
