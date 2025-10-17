import numpy as np

import torch
import torch.nn.functional as F

from ..decision_model import DecisionModelBase
from .psychophysical import PsychophysicalModel

from ..utils import glorot_uniform_init, setup_plotting
from ..torch_utils import select_array_inputs


class MixtureOfTheories(DecisionModelBase):
    def __init__(
        self,
        variant="full",
        util_func="GeneralPowerLossAverseUtil",
        pwf_func="KT_PWF",
        mixer_units=32,
        models=None,
        include_dom=True,
        inputs="outcomes",
        share_mixers=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.id = "MixtureOfSubjectiveFunctions"
        self.n_models = 2
        self.include_dom = include_dom
        if self.include_dom:
            self.requires_dom_mask = True
        self.variant = variant

        self.freeze_network = False
        self.inputs = inputs
        self.share_mixers = share_mixers

        if self.include_dom:
            self.prob_pick_dominated = 0.5

        self.models = {}
        self.models["EU_1"] = PsychophysicalModel(
            util_func=util_func,
            pwf="IdentityPWF",
            optimizer=None,
        )
        self.models["EU_2"] = PsychophysicalModel(
            util_func=util_func,
            pwf="IdentityPWF",
            optimizer=None,
        )
        self.models["SEU_1"] = PsychophysicalModel(
            util_func="IdentityUtil",
            pwf=pwf_func,
            optimizer=None,
        )
        self.models["SEU_2"] = PsychophysicalModel(
            util_func="IdentityUtil",
            pwf="IdentityPWF",
            optimizer=None,
        )

        self.n_models = 2

        input_size = 18 * 2
        if inputs in ["outcomes", "probabilities", "probs"]:
            input_size = 18
        elif inputs == "outcome_count":
            input_size = 1
        elif inputs == "outcome_counts":
            input_size = 2
        self.input_size = input_size

        n_units = mixer_units  # 20
        self.uf_mixer_params = {}
        # to connect inputs (both gambles) to hidden layer
        if input_size > 1:
            self.uf_mixer_params["uf_w1"] = glorot_uniform_init(n_units, input_size)
            self.uf_mixer_params["uf_b1"] = np.zeros(n_units)
        # to hidden layer to the convex weight output layer
        if input_size > 1:
            self.uf_mixer_params["uf_w2"] = glorot_uniform_init(self.n_models, n_units)
            self.uf_mixer_params["uf_b2"] = np.zeros(self.n_models)
        else:
            self.uf_mixer_params["uf_w2"] = glorot_uniform_init(self.n_models, 1)
            self.uf_mixer_params["uf_b2"] = np.zeros(self.n_models)
        
        # Cache for tensor versions of parameters on device
        self._uf_mixer_tensors = {}

        # if input_size > 1:
        #     # hack to get hidden activations
        #     def get_hidden(outcome):
        #         hidden = sigmoid(
        #             jnp.dot(self.uf_mixer_params["uf_w1"], outcome)
        #             + self.uf_mixer_params["uf_b1"]
        #         )
        #         return hidden

        # if input_size > 1:
        #     self.get_hidden = vmap(get_hidden)

        # CRITICAL PERFORMANCE FIX: Batch operations instead of per-sample
        def uf_mixer_batched(outcomes):
            # Use cached tensor parameters (already on device)
            w1 = self._uf_mixer_tensors.get("uf_w1")
            w2 = self._uf_mixer_tensors.get("uf_w2")
            b1 = self._uf_mixer_tensors.get("uf_b1")
            b2 = self._uf_mixer_tensors.get("uf_b2")
            
            # outcomes is already a tensor on correct device from select_array_inputs
            # Shape: (batch_size, input_size)
            if input_size > 1:
                # Batch matrix multiplication: (batch_size, input_size) @ (input_size, n_units)
                hidden = torch.sigmoid(
                    torch.matmul(outcomes, w1.T) + b1
                )
                # (batch_size, n_units) @ (n_units, n_models)
                output = F.softmax(
                    torch.matmul(hidden, w2.T) + b2,
                    dim=-1
                )
            else:
                # (batch_size, 1) @ (1, n_models)
                output = F.softmax(
                    torch.matmul(outcomes, w2.T) + b2,
                    dim=-1
                )
            return output
        
        self.uf_mixer = uf_mixer_batched

        self.pwf_mixer_params = {}
        # to connect inputs (both gambles) to hidden layer
        if input_size > 1:
            self.pwf_mixer_params["pwf_w1"] = glorot_uniform_init(n_units, input_size)
            self.pwf_mixer_params["pwf_b1"] = np.zeros(n_units)
        # to hidden layer to the convex weight output layer
        if input_size > 1:
            self.pwf_mixer_params["pwf_w2"] = glorot_uniform_init(
                self.n_models, n_units
            )
            self.pwf_mixer_params["pwf_b2"] = np.zeros(self.n_models)
        else:
            self.pwf_mixer_params["pwf_w2"] = glorot_uniform_init(self.n_models, 1)
            self.pwf_mixer_params["pwf_b2"] = np.zeros(self.n_models)
        
        # Cache for tensor versions of parameters on device
        self._pwf_mixer_tensors = {}

        if not self.share_mixers:
            # CRITICAL PERFORMANCE FIX: Batch operations instead of per-sample
            def pwf_mixer_batched(outcomes):
                # Use cached tensor parameters (already on device)
                w1 = self._pwf_mixer_tensors.get("pwf_w1")
                w2 = self._pwf_mixer_tensors.get("pwf_w2")
                b1 = self._pwf_mixer_tensors.get("pwf_b1")
                b2 = self._pwf_mixer_tensors.get("pwf_b2")
                
                # outcomes is already a tensor on correct device
                # Shape: (batch_size, input_size)
                if input_size > 1:
                    hidden = torch.sigmoid(
                        torch.matmul(outcomes, w1.T) + b1
                    )
                    output = F.softmax(
                        torch.matmul(hidden, w2.T) + b2,
                        dim=-1
                    )
                else:
                    output = F.softmax(
                        torch.matmul(outcomes, w2.T) + b2,
                        dim=-1
                    )
                return output

        else:
            # CRITICAL PERFORMANCE FIX: Batch operations, shared weights with uf_mixer
            def pwf_mixer_batched(outcomes):
                # Use cached tensor parameters (already on device) - shared with uf_mixer
                w1 = self._uf_mixer_tensors.get("uf_w1")
                w2 = self._pwf_mixer_tensors.get("pwf_w2")
                b1 = self._uf_mixer_tensors.get("uf_b1")
                b2 = self._pwf_mixer_tensors.get("pwf_b2")
                
                # outcomes is already a tensor on correct device
                # Shape: (batch_size, input_size)
                if input_size > 1:
                    hidden = torch.sigmoid(
                        torch.matmul(outcomes, w1.T) + b1
                    )
                    output = F.softmax(
                        torch.matmul(hidden, w2.T) + b2,
                        dim=-1
                    )
                else:
                    output = F.softmax(
                        torch.matmul(outcomes, w2.T) + b2,
                        dim=-1
                    )
                return output
        
        self.pwf_mixer = pwf_mixer_batched
        
        # Initialize tensor cache
        self._sync_mixer_tensors()

    def _sync_mixer_tensors(self):
        """Convert mixer parameters to tensors and cache them on device for efficiency"""
        # Cache uf_mixer tensors
        for key in ["uf_w1", "uf_w2", "uf_b1", "uf_b2"]:
            val = self.uf_mixer_params.get(key)
            if val is not None:
                if isinstance(val, torch.Tensor):
                    self._uf_mixer_tensors[key] = val.to(self.device)
                else:
                    self._uf_mixer_tensors[key] = torch.from_numpy(val).float().to(self.device)
        
        # Cache pwf_mixer tensors
        for key in ["pwf_w1", "pwf_w2", "pwf_b1", "pwf_b2"]:
            val = self.pwf_mixer_params.get(key)
            if val is not None:
                if isinstance(val, torch.Tensor):
                    self._pwf_mixer_tensors[key] = val.to(self.device)
                else:
                    self._pwf_mixer_tensors[key] = torch.from_numpy(val).float().to(self.device)

    def _infer_mixture_weights(self, dataset):
        """ for internal use only"""

        if isinstance(dataset, dict):
            outcomes, probabilities = dataset["outcomes"], dataset["probabilities"]
            if self.include_dom:
                self.A_dominated, self.B_dominated = dataset["dom_mask"]
        else:
            outcomes, probabilities = dataset.as_array(sort=self.required_sort)
            if self.include_dom:
                dataset.generate_dom_mask()
                self.A_dominated, self.B_dominated = dataset.dom_mask

        # the input to the mixer(s) is either one big, flat matrix
        # with all gamble pair information or subset we choose
        mixer_inputs = select_array_inputs(outcomes, probabilities, inputs=self.inputs, device=self.device)

        # infer the convex mixture weights
        uf_convex_weights = self.uf_mixer(mixer_inputs)
        pwf_convex_weights = self.pwf_mixer(mixer_inputs)

        return outcomes, probabilities, uf_convex_weights, pwf_convex_weights

    def infer_mixture_weights(self, dataset):
        """ End user can use this to extract mixture weights """
        _, _, uf_convex_weights, pwf_convex_weights = self._infer_mixture_weights(
            dataset
        )

        return {
            "uf_convex_weights": uf_convex_weights,
            "pwf_convex_weights": pwf_convex_weights,
        }

    def predict(self, dataset):

        (
            outcomes,
            probabilities,
            uf_convex_weights,
            pwf_convex_weights,
        ) = self._infer_mixture_weights(dataset)
        
        # Convert to torch if needed and move to device
        # (outcomes/probabilities should already be tensors from dataset.as_array)
        if not isinstance(outcomes, torch.Tensor):
            outcomes = torch.from_numpy(outcomes).float().to(self.device)
        elif outcomes.device != self.device:
            outcomes = outcomes.to(self.device)
            
        if not isinstance(probabilities, torch.Tensor):
            probabilities = torch.from_numpy(probabilities).float().to(self.device)
        elif probabilities.device != self.device:
            probabilities = probabilities.to(self.device)
        
        # uf_convex_weights and pwf_convex_weights should already be on device from mixer
        if not isinstance(uf_convex_weights, torch.Tensor):
            uf_convex_weights = torch.from_numpy(uf_convex_weights).float().to(self.device)
        elif uf_convex_weights.device != self.device:
            uf_convex_weights = uf_convex_weights.to(self.device)
            
        if not isinstance(pwf_convex_weights, torch.Tensor):
            pwf_convex_weights = torch.from_numpy(pwf_convex_weights).float().to(self.device)
        elif pwf_convex_weights.device != self.device:
            pwf_convex_weights = pwf_convex_weights.to(self.device)

        uf1_utils = self.models["EU_1"].utility_fn(outcomes)
        uf2_utils = self.models["EU_2"].utility_fn(outcomes)

        pwf1_weights = self.models["SEU_1"].weight_fn(probabilities)
        pwf2_weights = self.models["SEU_2"].weight_fn(probabilities)

        mixed_utils = (uf1_utils * uf_convex_weights[:, 0].reshape(-1, 1, 1)) + (
            uf2_utils * uf_convex_weights[:, 1].reshape(-1, 1, 1)
        )
        mixed_probs = (pwf1_weights * pwf_convex_weights[:, 0].reshape(-1, 1, 1)) + (
            pwf2_weights * pwf_convex_weights[:, 1].reshape(-1, 1, 1)
        )

        if self.variant == "full":
            mixed_predictions = torch.sum(mixed_utils * mixed_probs, axis=2)
        elif self.variant == "single_pwf":
            mixed_predictions = torch.sum(mixed_utils * pwf1_weights, axis=2)
        elif self.variant == "single_uf":
            mixed_predictions = torch.sum(uf1_utils * mixed_probs, axis=2)
        elif self.variant == "simulate_PT":
            mixed_predictions = torch.sum(uf1_utils * pwf1_weights, axis=2)

        mixed_predictions = self.decision_function(mixed_predictions)

        if self.include_dom:
            if not isinstance(self.B_dominated, torch.Tensor):
                B_dominated = torch.from_numpy(self.B_dominated).float().to(self.device)
                A_dominated = torch.from_numpy(self.A_dominated).float().to(self.device)
            elif self.B_dominated.device != self.device:
                B_dominated = self.B_dominated.to(self.device)
                A_dominated = self.A_dominated.to(self.device)
            else:
                # Already on correct device
                B_dominated = self.B_dominated
                A_dominated = self.A_dominated
                
            mixed_predictions = mixed_predictions * torch.vstack(
                [1 - B_dominated, 1 - B_dominated]
            ).T
            mixed_predictions = mixed_predictions * torch.vstack(
                [1 - A_dominated, 1 - A_dominated]
            ).T
            mixed_predictions = mixed_predictions + torch.vstack(
                [
                    B_dominated * self.prob_pick_dominated,
                    B_dominated * (1 - self.prob_pick_dominated),
                ]
            ).T
            mixed_predictions = mixed_predictions + torch.vstack(
                [
                    A_dominated * (1 - self.prob_pick_dominated),
                    A_dominated * self.prob_pick_dominated,
                ]
            ).T

        return mixed_predictions

    def set_params(self, params):
        # Store original set_params logic but also update cached tensors

        for model_key in self.models.keys():
            self.models[model_key].set_params(params[model_key])

        if not self.freeze_network:
            # get the params for the mixer network
            for mixer_params_key in self.uf_mixer_params.keys():
                self.uf_mixer_params[mixer_params_key] = params[mixer_params_key]

            for mixer_params_key in self.pwf_mixer_params.keys():
                self.pwf_mixer_params[mixer_params_key] = params[mixer_params_key]
            
            # Update cached tensors after parameters change
            self._sync_mixer_tensors()

        if self.include_dom:
            self.prob_pick_dominated = params["prob_pick_dominated"]

        self.T = params["T"]

    def get_params(self):

        params = {}

        # get params for each component model
        for model_key in self.models.keys():
            params[model_key] = self.models[model_key].get_params()

        if not self.freeze_network:
            # get the params for the mixer network
            for mixer_params_key in self.uf_mixer_params.keys():
                params[mixer_params_key] = self.uf_mixer_params[mixer_params_key]

            for mixer_params_key in self.pwf_mixer_params.keys():
                params[mixer_params_key] = self.pwf_mixer_params[mixer_params_key]

        if self.include_dom:
            params["prob_pick_dominated"] = self.prob_pick_dominated

        params["T"] = self.T

        return params

    def plot(self, show=True):

        plt = setup_plotting()

        uf1 = self.models["EU_1"].utility_fn
        uf2 = self.models["EU_2"].utility_fn
        pwf1 = self.models["SEU_1"].weight_fn
        pwf2 = self.models["SEU_2"].weight_fn

        fig, axes = plt.subplots(2, 2)

        uf1_ax, uf2_ax, pwf1_ax, pwf2_ax = (
            axes[0, 0],
            axes[1, 0],
            axes[0, 1],
            axes[1, 1],
        )

        uf1.plot(ax=uf1_ax), uf1_ax.set_title("UF 1")
        uf2.plot(ax=uf2_ax), uf2_ax.set_title("UF 2")
        pwf1.plot(ax=pwf1_ax), pwf1_ax.set_title("PWF 1")
        pwf2.plot(ax=pwf2_ax), pwf2_ax.set_title("PWF 2")

        if show:
            plt.show()
        else:
            return fig, axes
