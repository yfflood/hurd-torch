import numpy as np
import torch

from ..decision_model import DecisionModelBase

from ..initializer import initializer


class PsychophysicalModel(DecisionModelBase):
    def __init__(self, util_func="GeneralPowerLossAverseUtil", pwf="KT_PWF", **kwargs):
        super().__init__(**kwargs)
        self.id = "PsychophysicalModel"

        # initialize the utility, probability weighting, and loss parameters
        self.utility_fn = initializer(util_func, "utility")
        self.weight_fn = initializer(pwf, "pwf")

    def get_params(self):
        params = {}
        for key in self.utility_fn.parameters.keys():
            params["uf_" + key] = self.utility_fn.parameters[key]

        for key in self.weight_fn.parameters.keys():
            params["pwf_" + key] = self.weight_fn.parameters[key]

        # either get softmax temperature
        if self.stochastic_spec == "softmax":
            params["T"] = self.T
        # or get constant error term
        elif self.stochastic_spec == "constant-error":
            params["mu"] = self.mu

        return params

    def set_params(self, params):
        # set utility function params
        uf_params = {k[3:]: v for k, v in params.items() if k.startswith("u")}
        self.utility_fn.set_params(uf_params)

        pwf_params = {k[4:]: v for k, v in params.items() if k.startswith("p")}
        self.weight_fn.set_params(pwf_params)

        # either set softmax temperature
        if self.stochastic_spec == "softmax":
            self.T = params["T"]
        # or set constant error term
        elif self.stochastic_spec == "constant-error":
            self.mu = params["mu"]

    def predict(self, dataset):

        if isinstance(dataset, dict):
            outcomes, probabilities = dataset["outcomes"], dataset["probabilities"]
        else:
            outcomes, probabilities = dataset.as_array(sort=self.required_sort)
        
        # Convert to torch tensors if needed and move to device (only if not already on correct device)
        if not isinstance(outcomes, torch.Tensor):
            outcomes = torch.from_numpy(outcomes).float().to(self.device)
        elif outcomes.device != self.device:
            outcomes = outcomes.to(self.device)
        if not isinstance(probabilities, torch.Tensor):
            probabilities = torch.from_numpy(probabilities).float().to(self.device)
        elif probabilities.device != self.device:
            probabilities = probabilities.to(self.device)

        U, W = self.utility_fn, self.weight_fn

        outcomes, probabilities = U(outcomes), W(probabilities)

        utils = torch.sum(outcomes * probabilities, axis=2)

        return self.decision_function(utils)


class ExpectedUtilityModel(PsychophysicalModel):
    # will just be a wrapper with a fixed linear pwf
    # and takes only a specified utility function
    def __init__(self, util_func="GeneralPowerLossAverseUtil", **kwargs):
        super().__init__(
            util_func=util_func, pwf="IdentityPWF", **kwargs
        )
        self.id = "ExpectedUtilityModel"


class ExpectedValueModel(ExpectedUtilityModel):
    # will just be a wrapper with a fixed linear pwf
    def __init__(self, **kwargs):
        super().__init__(util_func="IdentityUtil", **kwargs)
        self.id = "ExpectedValueModel"


class ProspectTheoryModel(PsychophysicalModel):
    def __init__(self, util_func="GeneralPowerLossAverseUtil", pwf="KT_PWF", **kwargs):
        super().__init__(util_func=util_func, pwf=pwf, **kwargs)
        self.id = "ProspectTheoryModel"


class CumulativeProspectTheoryModel(PsychophysicalModel):
    def __init__(self, pwf_pos="KT_PWF", pwf_neg="KT_PWF", **kwargs):
        super().__init__(**kwargs)
        self.id = "CumulativeProspectTheoryModel"

        # we sort outcomes/probs based on ascending outcome value,
        # following Fennema & Wakker (1997)
        self.required_sort = "outcomes_asc"

        # initialize the two separate probability weighting functions
        self.weight_fn_pos = initializer(pwf_pos, "pwf")
        self.weight_fn_neg = initializer(pwf_neg, "pwf")

        # no single pwf for this class
        self.weight_fn = None

    def value_per_gamble(self, probs, outcomes, ld_mask):

        WP, WN = self.weight_fn_pos, self.weight_fn_neg

        def get_val():
            ld_p_cumsum = WN(torch.clamp(torch.cumsum(probs, dim=0), 0, 1))
            ld_p = torch.hstack([ld_p_cumsum[0].unsqueeze(0), ld_p_cumsum[1:] - ld_p_cumsum[:-1]])
            ld_p = ld_p * outcomes * ld_mask
            return ld_p

        ld_p = torch.where(ld_mask[0] != 0, get_val(), torch.zeros_like(outcomes))

        gd_mask = 1 - ld_mask
        gd_p_cumsum = WP(torch.clamp(torch.cumsum(probs * gd_mask, dim=0), 0, 1))
        gd_p = torch.hstack([gd_p_cumsum[0].unsqueeze(0), gd_p_cumsum[1:] - gd_p_cumsum[:-1]])
        gd_p = gd_p * outcomes * gd_mask

        return torch.sum(ld_p) + torch.sum(gd_p)

    def predict(self, dataset):

        if isinstance(dataset, dict):
            outcomes, probabilities = dataset["outcomes"], dataset["probabilities"]
        else:
            outcomes, probabilities = dataset.as_array(sort=self.required_sort)
        
        # Convert to torch if not already and move to device (only if not already on correct device)
        if not isinstance(outcomes, torch.Tensor):
            outcomes_np = outcomes
            outcomes = torch.from_numpy(outcomes).float().to(self.device)
        else:
            outcomes_np = outcomes.cpu().numpy()
            if outcomes.device != self.device:
                outcomes = outcomes.to(self.device)
        
        if not isinstance(probabilities, torch.Tensor):
            probabilities = torch.from_numpy(probabilities).float().to(self.device)
        elif probabilities.device != self.device:
            probabilities = probabilities.to(self.device)

        # most of the following to find first positive outcomes
        # do all searchsorts in advance
        def searchsort(x):
            return [np.searchsorted(x[0], 0), np.searchsorted(x[1], 0)]

        ld_masks = []
        for i in range(outcomes_np.shape[0]):
            first_pos = searchsort(outcomes_np[i])
            ld_masks.append(
                [
                    np.hstack(
                        [
                            np.ones(first_pos[0]),
                            np.zeros(outcomes_np.shape[2] - first_pos[0]),
                        ]
                    ),
                    np.hstack(
                        [
                            np.ones(first_pos[1]),
                            np.zeros(outcomes_np.shape[2] - first_pos[1]),
                        ]
                    ),
                ]
            )

        ld_masks = torch.from_numpy(np.array(ld_masks)).float().to(self.device)

        # pre-apply utility function
        outcomes = self.utility_fn(outcomes)

        # PyTorch vmap equivalent using list comprehension
        gambleA_values = torch.stack([
            self.value_per_gamble(probabilities[i, 0], outcomes[i, 0], ld_masks[i, 0])
            for i in range(probabilities.shape[0])
        ])

        gambleB_values = torch.stack([
            self.value_per_gamble(probabilities[i, 1], outcomes[i, 1], ld_masks[i, 1])
            for i in range(probabilities.shape[0])
        ])

        return self.decision_function(torch.vstack([gambleA_values, gambleB_values]).T)

    def get_params(self):
        params = {}
        for key in self.utility_fn.parameters.keys():
            params["uf_" + key] = self.utility_fn.parameters[key]

        for key in self.weight_fn_pos.parameters.keys():
            params["pos_pwf_" + key] = self.weight_fn_pos.parameters[key]

        for key in self.weight_fn_neg.parameters.keys():
            params["neg_pwf_" + key] = self.weight_fn_neg.parameters[key]

        # either get softmax temperature
        if self.stochastic_spec == "softmax":
            params["T"] = self.T
        # or get constant error term
        elif self.stochastic_spec == "constant-error":
            params["mu"] = self.mu

        return params

    def set_params(self, params):
        # set utility function params
        uf_params = {k[3:]: v for k, v in params.items() if k.startswith("u")}
        self.utility_fn.set_params(uf_params)

        # set the probability weighting function parameters
        pwf_pos = {k[8:]: v for k, v in params.items() if k.startswith("p")}
        pwf_neg = {k[8:]: v for k, v in params.items() if k.startswith("n")}
        self.weight_fn_pos.set_params(pwf_pos)
        self.weight_fn_neg.set_params(pwf_neg)

        # either set softmax temperature
        if self.stochastic_spec == "softmax":
            self.T = params["T"]
        # or set constant error term
        elif self.stochastic_spec == "constant-error":
            self.mu = params["mu"]


class TransferOfAttentionExchangeModel(DecisionModelBase):
    def __init__(self, **kwargs):
        """
        "Special" TAX model from:

            Birnbaum, M. H. (2008). New paradoxes of risky 
            decision making. Psychological Review, 115, 463–501.
        """
        super().__init__(**kwargs)
        self.id = "TransferOfAttentionExchange"

        # for TAX, outcomes must be ordered smallest to largest
        self.required_sort = "outcomes_asc"

        # single parameter for the power weighting function
        self.gamma = 0.5
        # single parameter for the power utility function
        self.beta = 0.5
        # configural weight parameter
        self.delta = 0.0

    def set_params(self, params):
        self.gamma = params["gamma"]
        self.beta = params["beta"]
        self.delta = params["delta"]

        self.T = params["T"]

    def get_params(self):
        params = {}
        params["gamma"] = self.gamma
        params["beta"] = self.beta
        params["delta"] = self.delta

        params["T"] = self.T
        return params

    def get_constraints(self):
        raise NotImplementedError

    def predict(self, dataset):

        if isinstance(dataset, dict):
            outcomes, probabilities = dataset["outcomes"], dataset["probabilities"]
        else:
            outcomes, probabilities = dataset.as_array(sort=self.required_sort)
        
        # Convert to torch if not already and move to device
        if not isinstance(outcomes, torch.Tensor):
            outcomes = torch.from_numpy(outcomes).float().to(self.device)
        else:
            outcomes = outcomes.to(self.device)
        if not isinstance(probabilities, torch.Tensor):
            probabilities = torch.from_numpy(probabilities).float().to(self.device)
        else:
            probabilities = probabilities.to(self.device)

        # apply power utility function
        signs = torch.sign(outcomes)
        absolute_outcomes = torch.abs(outcomes)
        utilities = signs * absolute_outcomes ** self.beta

        # apply power probability weighting function
        weighted_probs = probabilities ** self.gamma

        # first part of model is just subjective expected utility
        seu = torch.sum(utilities * weighted_probs, axis=2)

        # takes a single gamble and returns the TAX term
        def per_gamble_TAX_sum(outcomes, probs):

            # array length
            n = outcomes.numel()

            # mask for the real outcomes (excludes zero padding)
            non_padding_mask = (~((probs == 0.0) & (outcomes == 0.0))).float()
            # actual number of outcomes without zero-padding
            m = torch.sum(non_padding_mask)
            # helps remove the bad terms later due to the zero padding
            remove_mask = torch.outer(non_padding_mask, torch.ones(n - 1))[1:]

            # i and j outcomes in the tax model formula
            x_i, x_j = outcomes[1:], outcomes[:-1]

            x_i_outer = torch.outer(x_i, torch.ones(n - 1))
            x_j_outer = torch.outer(x_j, torch.ones(n - 1)).T

            xij_diffs = x_i_outer - x_j_outer
            # make sure to remove useless upper echelon
            xij_diffs = xij_diffs * torch.triu(torch.ones((n - 1, n - 1))).T

            # i and j probabilities in the tax model formula
            p_i, p_j = probs[1:], probs[:-1]

            # calculate both possible weights,
            # to be conditioned on delta
            pi_weights = (p_i * self.delta) / (m + 1)
            pj_weights = (p_j * self.delta) / (m + 1)

            # need to make them align with xij_diffs
            p_i_outer = torch.outer(pi_weights, torch.ones(n - 1))
            p_j_outer = torch.outer(pj_weights, torch.ones(n - 1)).T

            # selecting one weight set or the other
            delta_is_neg = float(self.delta < 0)
            weights = p_i_outer * delta_is_neg + p_j_outer * (1 - delta_is_neg)

            tax_sum_terms = xij_diffs * weights * remove_mask

            tax_sum = torch.sum(tax_sum_terms)

            return tax_sum

        # PyTorch vmap equivalent using list comprehension
        gambleA_tax_sums = torch.stack([
            per_gamble_TAX_sum(weighted_probs[i, 0], utilities[i, 0])
            for i in range(weighted_probs.shape[0])
        ])
        gambleB_tax_sums = torch.stack([
            per_gamble_TAX_sum(weighted_probs[i, 1], utilities[i, 1])
            for i in range(weighted_probs.shape[0])
        ])

        tax_sums = torch.vstack([gambleA_tax_sums, gambleB_tax_sums]).T

        final_utils = (seu + tax_sums) / torch.sum(weighted_probs, axis=2)

        return self.decision_function(final_utils)
