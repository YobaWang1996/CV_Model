import torch
import numpy as np
import time
from sklearn.metrics import roc_auc_score
from torch import optim
from abc import ABC, abstractmethod
from torch.utils.data import DataLoader


class BaseTrainer(ABC):
    """Trainer base class."""

    def __init__(self, optimizer_name: str, lr: float, n_epochs: int, lr_milestones: tuple,
                 weight_decay: float, device: str):
        super().__init__()
        self.optimizer_name = optimizer_name
        self.lr = lr
        self.n_epochs = n_epochs
        self.lr_milestones = lr_milestones
        self.weight_decay = weight_decay
        self.device = device

    @abstractmethod
    def train(self, dataset, dataset1, net, model_path):
        """
        Implement train method that trains the given network using the train_set of dataset.
        :return: Trained net
        """
        pass

    @abstractmethod
    def test(self, dataset, net):
        """
        Implement test method that evaluates the test_set of dataset on the given network.
        """
        pass


class DeepSVDDTrainer(BaseTrainer):

    def __init__(self, objective, R, c, nu: float, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 100,
                 lr_milestones: tuple = (), weight_decay: float = 1e-6, device: str = 'cuda'):

        super().__init__(optimizer_name, lr, n_epochs, lr_milestones, weight_decay, device)

        assert objective in ('one-class', 'soft-boundary'), "Objective must be either 'one-class' or 'soft-boundary'."
        self.objective = objective

        # Deep SVDD parameters
        self.R = torch.tensor(R, device=self.device)  # radius R initialized with 0 by default.
        self.c = torch.tensor(c, device=self.device) if c is not None else None
        self.nu = nu
        self.lr = lr

        # Optimization parameters
        self.warm_up_n_epochs = 0  # number of training epochs for soft-boundary Deep SVDD before radius R gets updated

        # Results
        self.train_time = None
        self.val_auc = None
        self.val_time = None
        self.test_auc = None
        self.test_time = None
        self.test_scores = None

    def train(self, dataset, val_data, net, model_path):

        # Set device for network
        net = net.to(self.device)

        # Set optimizer (Adam optimizer for now)
        for name, p in net.spatial_transformer.named_parameters():
            p.requires_grad = False

        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, net.parameters()),
                                lr=self.lr, betas=(0.9, 0.999), eps=1e-8,
                                weight_decay=self.weight_decay)

        # Set learning rate scheduler
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_milestones, gamma=0.1)

        # Initialize hypersphere center c (if c not loaded)
        if self.c is None:
            print('Initializing center c...')
            self.c = self.init_center_c(dataset, net)
            print('Center c initialized.')

        # Training
        print('Starting Training...')
        start_time = time.time()

        net.train()
        for epoch in range(self.n_epochs):

            loss_epoch = 0.0
            n_batches = 0
            epoch_start_time = time.time()
            for video_id in dataset.videos_ids():
                dataset.train_or_test(video_id)
                loader = DataLoader(dataset=dataset, drop_last=True)
                for _, data in enumerate(loader):
                    inputs, _ = data
                    inputs = torch.reshape(inputs, (-1, 3, 224, 224))
                    inputs = inputs.to(self.device)

                    # Zero the network parameter gradients
                    optimizer.zero_grad()

                    # Update network parameters via backpropagation: forward + backward + optimize
                    outputs = net(inputs)
                    dist = torch.sum((outputs - self.c) ** 2, dim=1)
                    if self.objective == 'soft-boundary':
                        scores = dist - self.R ** 2
                        loss = self.R ** 2 + (1 / self.nu) * torch.mean(torch.max(torch.zeros_like(scores), scores))
                    else:
                        loss = torch.mean(dist)
                    loss.backward()
                    optimizer.step()

                    # Update hypersphere radius R on mini-batch distances
                    if (self.objective == 'soft-boundary') and (epoch >= self.warm_up_n_epochs):
                        self.R.data = torch.tensor(get_radius(dist, self.nu), device=self.device)

                    loss_epoch += loss.item()
                    n_batches += 1

            scheduler.step()
            if epoch in self.lr_milestones:
                print('  LR scheduler: new learning rate is %g' % float(scheduler.get_last_lr()[0]))

            # log epoch statistics
            epoch_train_time = time.time() - epoch_start_time
            print('  Epoch {}/{}\t Time: {:.3f}\t Loss: {:.8f}'
                  .format(epoch + 1, self.n_epochs, epoch_train_time, loss_epoch / n_batches))

            #  Testing
            if (epoch+1) % 5 == 0:
                print('Starting Testing...')
                validate_time = time.time()
                th_auc = 0.
                label_all = []
                score_all = []

                net.eval()
                with torch.no_grad():
                    for video_id in val_data.videos_ids():
                        val_data.train_or_test(video_id)
                        val_loader = DataLoader(dataset=val_data)
                        for _, data in enumerate(val_loader):
                            inputs, labels = data
                            inputs = torch.reshape(inputs, (-1, 3, 224, 224))
                            inputs = inputs.to(self.device)
                            labels = labels.view(-1).tolist()[8]
                            outputs = net(inputs)
                            dist = torch.sum((outputs - self.c) ** 2, dim=1)
                            if self.objective == 'soft-boundary':
                                scores = dist - self.R ** 2
                            else:
                                scores = dist
                            scores = scores.cpu().tolist()
                            label_all.append(labels)
                            score_all.append(scores)

                    self.test_time = time.time() - validate_time
                    print('Testing time: %.3f' % self.test_time)

                    # Compute All Dataset AUC
                    label_list = np.array(label_all)
                    score_list = np.array(score_all)

                    self.test_auc = roc_auc_score(label_list, score_list)
                    print('Test set AUC: {:.2f}%'.format(100. * self.test_auc))

                    # EarlyStopping
                    if self.test_auc > th_auc:
                        net_dict = net.state_dict()
                        torch.save({'R': self.R,
                                    'c': self.c,
                                    'net_dict': net_dict}, model_path)
                        print("EarlyStopping！")
                        break

                print('Finished Testing...')

        self.train_time = time.time() - start_time
        print('Training time: %.3f' % self.train_time)

        print('Finished Training.')

        return net

    def test(self, dataset, net):

        # Set device for network
        net = net.to(self.device)

        # Testing
        print('Starting testing...')
        start_time = time.time()
        label = []
        score = []
        net.eval()
        with torch.no_grad():
            for video_id in dataset.videos_ids():
                dataset.train_or_test(video_id)
                test_loader = DataLoader(dataset=dataset)
                for _, data in enumerate(test_loader):
                    inputs, labels = data
                    inputs = torch.reshape(inputs, (-1, 3, 224, 224))
                    inputs = inputs.to(self.device)
                    labels = labels.view(-1).tolist()[8]
                    outputs = net(inputs)
                    dist = torch.sum((outputs - self.c) ** 2, dim=1)
                    if self.objective == 'soft-boundary':
                        scores = dist - self.R ** 2
                    else:
                        scores = dist
                    scores = scores.cpu().tolist()
                    # Save label, score in a list
                    label.append(labels)
                    score.append(scores)

        self.test_time = time.time() - start_time
        print('Testing time: %.3f' % self.test_time)

        self.test_scores = scores

        # Compute AUC
        labels = np.array(label)
        scores = np.array(score)

        self.test_auc = roc_auc_score(labels, scores)
        print('Test set AUC: {:.2f}%'.format(100. * self.test_auc))

        print('Finished testing.')

    def init_center_c(self, train_dataset, net, eps=0.001):
        """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
        n_samples = 0
        c = torch.zeros(net.outdim, device=self.device)

        net.eval()
        with torch.no_grad():
            for video_id in train_dataset.videos_ids():
                train_dataset.train_or_test(video_id)
                train_loader = DataLoader(dataset=train_dataset)
                for _, data in enumerate(train_loader):
                    # get the inputs of the batch
                    inputs, _ = data
                    inputs = torch.reshape(inputs, (-1, 3, 224, 224))
                    inputs = inputs.to(self.device)
                    outputs = net(inputs)
                    n_samples += outputs.shape[0]
                    c += torch.sum(outputs, dim=0)

        c /= n_samples

        # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps

        return c


def get_radius(dist: torch.Tensor, nu: float):
    """Optimally solve for radius R via the (1-nu)-quantile of distances."""
    return np.quantile(np.sqrt(dist.clone().data.cpu().numpy()), 1 - nu)
