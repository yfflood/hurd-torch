import os, json
from copy import deepcopy
from datetime import datetime
from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn.functional as F

from .utils import mkdir
from .initializer import initializer
from .optimizers import SGD


class DecisionModelBase(ABC):
    def __init__(
        self,
        loss_function="mse",
        stochastic_spec="softmax",  # "softmax" (logit) or "constant-error"
        optimizer=None,
        verbose=2,
        device=None,
    ):
        self.id = "DecisionModelBase"
        self.stochastic_spec = stochastic_spec
        self.loss_function = initializer(loss_function, "loss")
        self.optimizer = optimizer
        self.requires_dom_mask = False
        self.requires_ld_mask = False
        self.verbose = verbose
        self.has_validation_data = False
        # assume no data sorting is needed until specified
        self.required_sort = "none"
        
        # Set device (cuda if available and not specified, else cpu)
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        if self.verbose > 0:
            print(f"Using device {self.device} for model {self.id}")

        # either we send utils into a softmax with temperature
        # or we use a constant error term for strict (binary) preferences
        if self.stochastic_spec == "softmax":
            # same as "logit"
            self.T = 1.0

            def softmax_error(utils, T):
                if not isinstance(utils, torch.Tensor):
                    utils = torch.from_numpy(utils).float()
                # T might be a tensor parameter, ensure utils is on same device
                if isinstance(T, torch.Tensor):
                    utils = utils.to(T.device)
                return F.softmax(utils * T, dim=-1)

            self.stochastic_func = softmax_error

        elif self.stochastic_spec == "constant-error":
            self.mu = 0.0

            def constant_error(utils, mu):
                # (PAGE 3) He, L., Zhao, W. J., & Bhatia, S. (2020).
                # An ontology of decision models. Psychological Review.

                # just a torch compatible way to set binary
                # decisions to 1-(mu/2) and mu/2
                # ex: [pick_A, pick_B] -> [0, 1] -> ...
                #     prob_pick_A, prob_pick_B] -> [0.1, 0.9]
                # utils are first turned into binary decisions
                if not isinstance(utils, torch.Tensor):
                    utils = torch.from_numpy(utils).float()
                # Preserve gradient if mu is already a tensor
                mu_t = mu if isinstance(mu, torch.Tensor) else torch.tensor(mu, dtype=torch.float32)
                # Ensure utils is on same device as mu_t
                if isinstance(mu_t, torch.Tensor):
                    utils = utils.to(mu_t.device)
                error = torch.sigmoid(mu_t) / 2.0
                p_a = (utils[:, 0] >= utils[:, 1]).float() * (1.0 - error)
                p_a += (utils[:, 0] < utils[:, 1]).float() * error
                p_b = 1 - p_a
                return torch.vstack([p_a, p_b]).T

            self.stochastic_func = constant_error

    def __str__(self):
        kvs = ", ".format(["{}={}".format(k, v) for k, v in self.config.items()])
        return "{}({})".format(self.id, kvs)

    def save_params(self, path):
        with open(path, "w") as handle:
            json.dump(self.get_params(), handle, indent=4, sort_keys=True)

    @abstractmethod
    def set_params(self):
        pass

    @abstractmethod
    def get_params(self):
        pass

    @abstractmethod
    def predict(self):
        pass

    def decision_function(self, utils):

        if self.stochastic_spec == "softmax":
            return self.stochastic_func(utils, self.T)

        elif self.stochastic_spec == "constant-error":
            return self.stochastic_func(utils, self.mu)

    def evaluate(self, dataset):

        if isinstance(dataset, dict):
            targets = dataset["targets"]
        else:
            targets = dataset.cached_arrays[2]

        preds = self.predict(dataset=dataset)

        return self.loss_function(preds, targets)

    def compute_accuracy(self, dataset=None):

        # this function gives us a sense of how well model predicts
        # binary decision preferences, disregarding exact choice rates

        _, _, targets = dataset.cached_arrays

        preds = self.predict(dataset=dataset)
        b_preds = preds[:, 1]
        
        # Convert to numpy if torch tensors
        if isinstance(b_preds, torch.Tensor):
            b_preds = b_preds.cpu().detach().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().detach().numpy()
            
        return np.sum((b_preds > 0.5) == (targets > 0.5)) / b_preds.size

    def fit(self, dataset, val_dataset=None, batch_size=None):
        self.dataset = dataset
        self.val_dataset = val_dataset

        if self.val_dataset:
            self.has_validation_data = True

        # make sure the model has an optimizer
        if not self.optimizer:
            print("No optimizer specified. Using default.")
            self.optimizer = SGD()

        self.optimizer.initialize(self, batch_size=batch_size)

        if self.optimizer.tolerance:
            n_iters_no_progress = 0

        for ix in range(self.optimizer.n_iters):

            step_results = self.optimizer.step(ix)

            if self.optimizer.tolerance:
                if not step_results["improvement"]:
                    n_iters_no_progress += 1
                    if n_iters_no_progress >= self.optimizer.patience:
                        break
                else:
                    n_iters_no_progress = 0

            self.results = {
                "best_params": deepcopy(self.optimizer.best_params),
                "best_loss": self.optimizer.best_loss,
            }
