import torch
from space_time_features import spatial_temporal_Fusion_Module
from deepSVDD_trainer import DeepSVDDTrainer


class DeepSVDD(object):
    """A class for the Deep SVDD method.

    Attributes:
        objective: A string specifying the Deep SVDD objective (either 'one-class' or 'soft-boundary').
        nu: Deep SVDD hyperparameter nu (must be 0 < nu <= 1).
        R: Hypersphere radius R.
        c: Hypersphere center c.
        net: The neural network.
        trainer: DeepSVDDTrainer to train a Deep SVDD model.
        optimizer_name: A string indicating the optimizer to use for training the Deep SVDD network.
        results: A dictionary to save the results.
    """

    def __init__(self, objective: str = 'one-class', nu: float = 0.01):
        """Inits DeepSVDD with one of the two objectives and hyperparameter nu."""

        assert objective in ('one-class', 'soft-boundary'), "Objective must be either 'one-class' or 'soft-boundary'."
        self.objective = objective
        assert (0 < nu) & (nu <= 1), "For hyperparameter nu, it must hold: 0 < nu <= 1."
        self.nu = nu
        self.R = 0.0  # hypersphere radius R
        self.c = None  # hypersphere center c

        self.net = None  # neural network

        self.trainer = None
        self.optimizer_name = None

        self.results = {
            'train_time': None,
            'test_auc': None,
            'test_time': None,
            'test_scores': None,
        }

    def set_network(self, n_frame):
        """Builds the neural network."""
        self.net = spatial_temporal_Fusion_Module(num_frames=n_frame)

    def train(self, dataset, val_data, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 50,
              lr_milestones: tuple = (), weight_decay: float = 1e-6, device: str = 'cuda', model_path: str = ""):
        """Trains the Deep SVDD model on the training data."""

        self.trainer = DeepSVDDTrainer(self.objective, self.R, self.c, self.nu, optimizer_name, lr=lr,
                                       n_epochs=n_epochs, lr_milestones=lr_milestones,
                                       weight_decay=weight_decay, device=device)
        # Get the model
        self.net = self.trainer.train(dataset, val_data, self.net, model_path)

        self.R = float(self.trainer.R.cpu().data.numpy())  # get float
        self.c = self.trainer.c.cpu().data.numpy().tolist()  # get list
        self.results['train_time'] = self.trainer.train_time

    def test(self, dataset, device: str = 'cuda'):
        """Tests the Deep SVDD model on the test data."""

        if self.trainer is None:
            self.trainer = DeepSVDDTrainer(self.objective, self.R, self.c, self.nu,
                                           device=device)

        self.trainer.test(dataset, self.net)
        # Get results
        self.results['test_auc'] = self.trainer.test_auc
        self.results['test_time'] = self.trainer.test_time
        self.results['test_scores'] = self.trainer.test_scores

    def save_model(self, export_model):
        """Save Deep SVDD model to export_model."""

        net_dict = self.net.state_dict()

        torch.save({'R': self.R,
                    'c': self.c,
                    'net_dict': net_dict}, export_model)
