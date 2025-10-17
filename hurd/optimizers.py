import time, math
from copy import deepcopy
from random import random, gauss
from abc import ABC, abstractmethod

try:
    from tqdm.notebook import tqdm
except:
    from tqdm import tqdm

from functools import partial

import torch
import numpy as np

from hurd.utils import dict2str, fix_torch_dict_floats


class OptimizerBase(ABC):
    def __init__(self):
        self.id = "OptimizerBase"
        self.model = None

        self.best_params = {}
        self.best_loss = np.inf
        self.best_training_loss = np.inf

        self.train_loss_history = []
        self.val_loss_history = []
        self.param_history = []
        self.grad_history = []

    def check_progress(self, ix, n, curr_loss, train_loss, t0=None):

        if self.model.has_validation_data:
            val_result_string = ", Val Loss: {:.5f}".format(curr_loss)
        else:
            val_result_string = ""

        result_str = "[Epoch {}/{}] Train Loss: {:.5f}{}, Elapsed: {:.2f}s".format(
            ix + 1, n, train_loss, val_result_string, time.time() - t0
        )

        if self.tolerance:
            found_improvement = not math.isclose(
                curr_loss, self.best_loss, abs_tol=self.tolerance
            )
            found_improvement = found_improvement and (curr_loss < self.best_loss)
        else:
            found_improvement = True

        if curr_loss < self.best_loss:
            self.best_params = deepcopy(self.model.get_params())
            self.best_loss = curr_loss
            result_str += " * New Best * "

        if train_loss < self.best_training_loss:
            self.best_training_loss = train_loss

        if self.model.verbose > 1:
            print(result_str, flush=True)

        return found_improvement

    def finish(self):
        # report the best model fit and parameters
        if self.model.verbose > -1:
            fstr = "Final best model - Loss: {:.4f}, Params: {}"
            print(fstr.format(self.best_loss, dict2str(self.best_params)))


class GradientBasedOptimizer(OptimizerBase):
    def __init__(
        self,
        alg="sgd",
        lr=0.1,
        n_iters=10,
        use_jit=True,  # Kept for compatibility but not used in PyTorch
        tol=None,
        patience=50,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.id = "GradientBasedOptimizer"
        self.lr = lr
        self.n_iters = n_iters
        self.use_jit = use_jit  # Kept for compatibility
        self.batch_size = None
        self.tolerance = tol
        self.patience = patience
        self.alg = alg # update algorithm (sgd or adam)
        self.alg_args = {} # argument inputs for alg

    def initialize(self, model, batch_size=None):

        self.model = model
        self.batch_size = batch_size

        # Convert model parameters to PyTorch tensors
        params = fix_torch_dict_floats(deepcopy(self.model.get_params()))
        
        # Get model device
        device = self.model.device
        
        # Create list of torch parameters that require gradients
        self.param_dict = {}
        for key, val in params.items():
            if isinstance(val, dict):
                # Nested dictionary (for mixture models)
                self.param_dict[key] = {}
                for subkey, subval in val.items():
                    if isinstance(subval, (list, np.ndarray)):
                        # Handle arrays/lists
                        if isinstance(subval, list):
                            subval = np.array(subval)
                        tensor = torch.from_numpy(subval).float().to(device).requires_grad_(True)
                    else:
                        tensor = torch.tensor(float(subval), device=device, requires_grad=True)
                    self.param_dict[key][subkey] = tensor
            elif isinstance(val, (list, np.ndarray)):
                # Handle arrays/lists
                if isinstance(val, list):
                    val = np.array(val)
                tensor = torch.from_numpy(val).float().to(device).requires_grad_(True)
                self.param_dict[key] = tensor
            else:
                self.param_dict[key] = torch.tensor(float(val), device=device, requires_grad=True)
        
        self.best_params = deepcopy(params)
        
        # Collect all tensors that require gradients
        self.torch_params = []
        def collect_tensors(d):
            for key, val in d.items():
                if isinstance(val, dict):
                    collect_tensors(val)
                elif isinstance(val, torch.Tensor) and val.requires_grad:
                    self.torch_params.append(val)
        collect_tensors(self.param_dict)
        
        # Create PyTorch optimizer
        if self.alg == "sgd":
            self.optimizer = torch.optim.SGD(self.torch_params, lr=self.lr)
        elif self.alg == "adam":
            b1 = self.alg_args.get('b1', 0.9)
            b2 = self.alg_args.get('b2', 0.999)
            eps = self.alg_args.get('eps', 1e-8)
            self.optimizer = torch.optim.Adam(self.torch_params, lr=self.lr, 
                                             betas=(b1, b2), eps=eps)
        else:
            raise ValueError(f"Unknown optimizer algorithm: {self.alg}")
        
        # Define loss functions
        def get_train_loss():
            params = self._extract_params_from_tensors()
            self.model.set_params(params)
            loss = self.model.evaluate(self.model.dataset)
            # Convert to tensor if needed, but keep gradients
            if not isinstance(loss, torch.Tensor):
                loss = torch.tensor(loss, dtype=torch.float32)
            return loss

        def get_batch_loss(batch):
            params = self._extract_params_from_tensors()
            self.model.set_params(params)
            loss = self.model.evaluate(batch)
            # Convert to tensor if needed, but keep gradients
            if not isinstance(loss, torch.Tensor):
                loss = torch.tensor(loss, dtype=torch.float32)
            return loss

        if self.model.has_validation_data:
            def get_val_loss():
                with torch.no_grad():
                    params = self._extract_params_from_tensors()
                    self.model.set_params(params)
                    loss = self.model.evaluate(self.model.val_dataset)
                    if isinstance(loss, torch.Tensor):
                        return loss.item()
                    return loss
        else:
            def get_val_loss():
                with torch.no_grad():
                    params = self._extract_params_from_tensors()
                    self.model.set_params(params)
                    loss = self.model.evaluate(self.model.dataset)
                    if isinstance(loss, torch.Tensor):
                        return loss.item()
                    return loss

        self.get_train_loss = get_train_loss
        self.get_batch_loss = get_batch_loss
        self.get_val_loss = get_val_loss
    
    def _extract_params_from_tensors(self, detach=False):
        """Extract parameter values from torch tensors back to dict format
        
        Args:
            detach: If True, detach tensors from computation graph (for saving/logging)
                   If False, keep tensors attached for gradient computation
        """
        params = {}
        for key, val in self.param_dict.items():
            if isinstance(val, dict):
                params[key] = {}
                for subkey, subval in val.items():
                    if isinstance(subval, torch.Tensor):
                        if detach:
                            if subval.ndim == 0:
                                params[key][subkey] = subval.item()
                            else:
                                params[key][subkey] = subval.detach().cpu().numpy()
                        else:
                            # Keep tensor with gradients
                            params[key][subkey] = subval
                    else:
                        params[key][subkey] = subval
            elif isinstance(val, torch.Tensor):
                if detach:
                    if val.ndim == 0:
                        params[key] = val.item()
                    else:
                        params[key] = val.detach().cpu().numpy()
                else:
                    # Keep tensor with gradients
                    params[key] = val
            else:
                params[key] = val
        return params

    def step(self, ix):

        t0 = time.time()

        if self.batch_size:

            n_batches = len(self.model.dataset) / self.batch_size

            for (bix, batch) in enumerate(
                tqdm(
                    self.model.dataset.iter_batch(batch_size=self.batch_size),
                    total=n_batches,
                )
            ):
                batch_arrays = batch.as_array(
                    sort=self.model.required_sort, return_targets=True
                )
                batch_dict = {
                    "outcomes": batch_arrays[0],
                    "probabilities": batch_arrays[1],
                    "targets": batch_arrays[2],
                }
                if self.model.requires_dom_mask:
                    batch.generate_dom_mask()
                    batch_dict["dom_mask"] = batch.dom_mask

                # PyTorch optimization step
                self.optimizer.zero_grad()
                loss = self.get_batch_loss(batch_dict)
                loss.backward()
                self.optimizer.step()

        else:
            # Full batch optimization
            self.optimizer.zero_grad()
            loss = self.get_train_loss()
            loss.backward()
            self.optimizer.step()

        # Extract updated parameters (detached for evaluation)
        updated_params = self._extract_params_from_tensors(detach=True)
        self.model.set_params(updated_params)

        # Evaluate losses (no gradients needed)
        with torch.no_grad():
            train_params = self._extract_params_from_tensors(detach=True)
            self.model.set_params(train_params)
            train_loss = self.model.evaluate(self.model.dataset)
            if isinstance(train_loss, torch.Tensor):
                train_loss = train_loss.item()
            
            val_loss = self.get_val_loss()
            if isinstance(val_loss, torch.Tensor):
                val_loss = val_loss.item()

        # Collect gradients for history
        grads = {}
        for key, val in self.param_dict.items():
            if isinstance(val, dict):
                grads[key] = {}
                for subkey, subval in val.items():
                    if isinstance(subval, torch.Tensor) and subval.grad is not None:
                        if subval.ndim == 0:
                            grads[key][subkey] = subval.grad.item()
                        else:
                            grads[key][subkey] = subval.grad.detach().cpu().numpy()
            elif isinstance(val, torch.Tensor) and val.grad is not None:
                if val.ndim == 0:
                    grads[key] = val.grad.item()
                else:
                    grads[key] = val.grad.detach().cpu().numpy()

        # update optimization history
        self.train_loss_history.append(float(train_loss))
        self.val_loss_history.append(float(val_loss))
        self.param_history.append(updated_params)  # Already detached/converted
        self.grad_history.append(fix_torch_dict_floats(grads))

        # evaluate the model, check for improvement, report speed
        is_improvement = self.check_progress(
            ix, self.n_iters, val_loss, train_loss, t0=t0
        )

        return {
            "epoch": ix,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "improvement": is_improvement,
        }


class SGD(GradientBasedOptimizer):
    def __init__(self, **kwargs):
        super().__init__(alg="sgd", **kwargs)
        self.id = "SGD"


class Adam(GradientBasedOptimizer):
    def __init__(self, b1=0.9, b2=0.999, eps=1e-08, **kwargs):
        super().__init__(alg="adam", **kwargs)
        self.id = "Adam"
        self.alg_args = {"b1": b1, "b2": b2, "eps": eps}
