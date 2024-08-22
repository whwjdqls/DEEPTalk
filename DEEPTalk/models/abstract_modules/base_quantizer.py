import torch
from torch import nn
from abc import ABC, abstractmethod


class BaseVectorQuantizer(ABC, nn.Module):

    def __init__(self, num_embeddings: int, embedding_dim: int):

        """
        :param num_embeddings: size of the latent dictionary (num of embedding vectors).
        :param embedding_dim: size of a single tensor in dict.
        """

        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # create the codebook of the desired size
        self.codebook = nn.Embedding(self.num_embeddings, self.embedding_dim)

        # wu/decay init (may never be used)
        self.kl_warmup = None
        self.temp_decay = None

    def init_codebook(self) -> None:
        """
        uniform initialization of the codebook
        """
        nn.init.uniform_(self.codebook.weight, -1 / self.num_embeddings, 1 / self.num_embeddings)

    @abstractmethod
    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.IntTensor, float):
        """
        :param x: tensors (output of the Encoder - B,D,H,W).
        :return quantized_x (B, D, H, W), detached codes (B, H*W), latent_loss
        """
        pass

    @abstractmethod
    def vec_to_codes(self, x: torch.Tensor) -> torch.IntTensor:
        """
        :param x: tensors (output of the Encoder - B,D,H,W).
        :return flat codebook indices (B, H * W)
        """
        pass

    @torch.no_grad()
    def get_codebook(self) -> torch.nn.Embedding:
        return self.codebook.weight

    @torch.no_grad()
    def codes_to_vec(self, codes: torch.IntTensor) -> torch.Tensor:
        """
        :param codes: int tensors to decode (B, N).
        :return flat codebook indices (B, N, D)
        """

        quantized = self.get_codebook()[codes]
        return quantized

    def get_codebook_usage(self, index_count: torch.Tensor):
        """
        :param index_count: (n, ) where n is the codebook size, express the number of times each index have been used.
        :return: prob of each index to be used: (n, ); perplexity: float; codebook_usage: float 0__1
        """

        # get used idx as probabilities
        used_indices = index_count / torch.sum(index_count)

        # perplexity
        perplexity = torch.exp(-torch.sum(used_indices * torch.log(used_indices + 1e-10), dim=-1)).sum().item()

        # get the percentage of used codebook
        n = index_count.shape[0]
        used_codebook = (torch.count_nonzero(used_indices).item() * 100) / n

        return used_indices, perplexity, used_codebook

    @torch.no_grad()
    def reinit_unused_codes(self, codebook_usage: torch.Tensor):
        """
        Re-initialize unused vectors according to the likelihood of used ones.
        :param codebook_usage: (n, ) where n is the codebook size, distribution probability of codebook usage.
        """

        device = codebook_usage.device
        n = codebook_usage.shape[0]

        # compute unused codes
        unused_codes = torch.nonzero(torch.eq(codebook_usage, torch.zeros(n, device=device))).squeeze(1)
        n_unused = unused_codes.shape[0]

        # sample according to most used codes.
        torch.use_deterministic_algorithms(False,warn_only=True)
        replacements = torch.multinomial(codebook_usage, n_unused, replacement=True)
        torch.use_deterministic_algorithms(True,warn_only=True)

        # update unused codes
        new_codes = self.codebook.weight[replacements]
        self.codebook.weight[unused_codes] = new_codes