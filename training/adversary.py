import torch

class Adversary:
    def __init__(self, model, loss_fn, epsilon):
        self.model = model
        self.loss_fn = loss_fn
        self.epsilon = epsilon

    def train_adversarial(self, images, optimizer, adv_steps):
        self.model.train()
        for i in range(adv_steps):
            optimizer.zero_grad()
            images = self.attacker.attack(images)

        preds = self.model(images)
        loss = self.loss_fn(preds, labels)
        loss.backward()
        optimizer.step()
