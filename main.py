import torch
from deepSVDD import DeepSVDD
from UCSDPed2 import UCSDPed2
import numpy as np


def set_seed(seed):
    """
    :param seed: Set fixed random seeds,In order to ensure the repeatability of the model.
    """

    torch.manual_seed(seed)  # CPU
    torch.cuda.manual_seed(seed)  # GPU
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True


def main(model_path):
    """
    Deep SVDD, a fully deep method for anomaly detection.
    """

    # device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load data
    path = ""  # data path

    train_dataset = UCSDPed2(path,
                             train_test='Train',
                             number_frame=16,
                             step=8)

    test_dataset = UCSDPed2(path,
                            train_test="Test",
                            number_frame=16,
                            step=8)

    # Initialize DeepSVDD model and set network
    deep_SVDD = DeepSVDD(objective='soft-boundary', nu=0.01)  # one-class/soft-boundary
    deep_SVDD.set_network(n_frame=16)

    # Train model on datasets
    deep_SVDD.train(dataset=train_dataset,
                    val_data=test_dataset,
                    optimizer_name="adamW",
                    lr=1e-4,
                    n_epochs=200,
                    lr_milestones=(),
                    weight_decay=1e-5,
                    device=device,
                    model_path=model_path)

    # Test model
    deep_SVDD.test(dataset=test_dataset, device=device)


if __name__ == '__main__':
    set_seed(10000)
    save_path = ""
    main(save_path)
