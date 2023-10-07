import torch
import numpy as np
from typing import Union
from abc import ABC, abstractmethod


class Sampler(ABC):
    @abstractmethod
    def generate_data(self, n_samples: int) -> tuple[torch.Tensor, torch.Tensor]:
        pass


class DummyData(Sampler):
    def __init__(self, x_size):
        self.x_size = x_size

    def generate_data(self, n_samples: int) -> tuple[torch.Tensor, torch.Tensor]:
        X = torch.rand((n_samples, self.x_size))
        y = torch.mean(X, axis=1) > 0.5
        return X, y


class MultitaskSparseParity(Sampler):
    n_bits: int
    n_data_bits: int  # Size of data vector
    n_control_bits: int  # Number of tasks
    alpha: float  # Sparsity parameter
    k: int  # Number of bits used in each task mask.

    def __init__(self, n_control_bits, n_data_bits, k, alpha):
        self.n_data_bits = n_data_bits
        self.n_control_bits = n_control_bits
        self.alpha = alpha
        self.k = k
        self.bits = n_data_bits + n_control_bits

        self.task_masks = self._n_bit_mask(self.n_control_bits, self.n_data_bits, k)

    def generate_data(
        self, n_samples: int, force_task: Union[int, None] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ## Generate data bits.
        x_data = torch.randint(
            2, (n_samples, self.n_data_bits)
        )  # , names=("n_samples", "n_data"))

        ## Generate task bits
        if force_task is None:
            probs = np.array(
                [x ** (-(self.alpha + 1)) for x in range(1, self.n_control_bits + 1)]
            )
            probs = probs / sum(probs)
            x_task_index = np.random.choice(
                self.n_control_bits, size=n_samples, p=probs
            )
        else:
            x_task_index = np.array([force_task] * n_samples)
        # One hot encode task indicies
        x_control = self._one_hot_encode(x_task_index)
        # plt.hist(x_task_index, bins= self.n_control_bits)

        ## Combine task and data bits.
        X = torch.cat((torch.tensor(x_control), torch.tensor(x_data)), axis=1)

        ## Calculate task outputs.
        y = torch.remainder(
            torch.sum(self.task_masks[x_task_index] * x_data, axis=1), 2
        )
        return X, y

    def _one_hot_encode(self, task_indices):
        n_samples = task_indices.shape[0]
        result = np.zeros(shape=(n_samples, self.n_control_bits))
        result[np.arange(n_samples), task_indices] = 1
        return result

    def _n_bit_mask(self, n_samples: int, n_possible: int, n_set: int):
        rand_mat = torch.rand(n_samples, n_possible)
        k_th_quant = torch.topk(rand_mat, n_set, largest=False)[0][:, -1:]
        bool_tensor = rand_mat <= k_th_quant
        desired_tensor = torch.where(bool_tensor, torch.tensor(1), torch.tensor(0))
        return desired_tensor
