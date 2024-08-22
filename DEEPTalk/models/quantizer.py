## Code adapted from [Esser, Rombach 2021]: https://compvis.github.io/taming-transformers/

import torch
import torch.nn as nn
import torch.nn.functional as F
from .abstract_modules.base_quantizer import BaseVectorQuantizer
from einops import rearrange, einsum
class VectorQuantizer(nn.Module):
    """
    see https://github.com/MishaLaskin/vqvae/blob/d761a999e2267766400dc646d82d3ac3657771d4/models/quantizer.py
    ____________________________________________
    Discretization bottleneck part of the VQ-VAE.
    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    _____________________________________________
    """

    def __init__(self, n_e, e_dim, beta):
        super(VectorQuantizer, self).__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def forward(self, z):
        """
        Inputs the output of the encoder network z and maps it to a discrete
        one-hot vector that is the index of the closest embedding vector e_j
        z (continuous) -> z_q (discrete)
        z.shape = (batch, channel, height, width)
        quantization pipeline:
            1. get encoder input (B,C,H,W)
            2. flatten input to (B*H*W,C)
        """
        # reshape z -> (batch, height, width, channel) and flatten
        #print('zshape', z.shape)
        z = z.permute(0, 2, 1).contiguous() # input is (B,T/q,dim) -> (B,dim,T/q)
        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())

        ## could possible replace this here
        # #\start...
        # find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)

        min_encodings = torch.zeros(
            min_encoding_indices.shape[0], self.n_e).to(z)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # dtype min encodings: torch.float32
        # min_encodings shape: torch.Size([2048, 512])
        # min_encoding_indices.shape: torch.Size([2048, 1])

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)
        #.........\end

        # with:
        # .........\start
        #min_encoding_indices = torch.argmin(d, dim=1)
        #z_q = self.embedding(min_encoding_indices)
        # ......\end......... (TODO)

        # compute loss for embedding
        loss = self.beta * torch.mean((z_q.detach()-z)**2) + \
                   torch.mean((z_q - z.detach()) ** 2)
        #loss = torch.mean((z_q.detach()-z)**2) + self.beta * \
        #    torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # perplexity
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        # reshape back to match original input shape
        z_q = z_q.permute(0, 2, 1).contiguous()

        return z_q, loss, (perplexity, min_encodings, min_encoding_indices)

    def get_distance(self, z):
        z = z.permute(0, 2, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())
        d = torch.reshape(d, (z.shape[0], -1, z.shape[2])).permute(0,2,1).contiguous()
        return d

    def get_codebook_entry(self, indices, shape):
        # shape specifying (batch, height, width, channel)
        # TODO: check for more easy handling with nn.Embedding
        min_encodings = torch.zeros(indices.shape[0], self.n_e).to(indices)
        min_encodings.scatter_(1, indices[:,None], 1)

        # get quantized latent vectors
        #print(min_encodings.shape, self.embedding.weight.shape)
        z_q = torch.matmul(min_encodings.float(), self.embedding.weight)

        if shape is not None:
            z_q = z_q.view(shape)

            # reshape back to match original input shape
            #z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q
    
    

class EMAVectorQuantizer(BaseVectorQuantizer):
    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float = 0.25, decay: float = 0.95,
                 epsilon: float = 1e-5):
        """
        EMA ALGORITHM
        Each codebook entry is updated according to the encoder outputs who selected it.
        The important thing is that the codebook updating is not a loss term anymore.
        Specifically, for every codebook item wi, the mean code mi and usage count Ni are tracked:
        Ni ← Ni · γ + ni(1 − γ),
        mi ← mi · γ + Xnij e(xj )(1 − γ),
        wi ← mi Ni
        where γ is a discount factor

        :param num_embeddings: size of the latent dictionary (num of embedding vectors).
        :param embedding_dim: size of a single tensor in dictionary
        :param commitment_cost: scaling factor for e_loss
        :param decay: decay for EMA updating
        :param epsilon: smoothing parameters for EMA weights
        """
        super().__init__(num_embeddings, embedding_dim)
        
        self.commitment_cost = commitment_cost
        # EMA does not require grad
        self.codebook.requires_grad_(False)
        # ema parameters
        # ema usage count: total count of each embedding trough epochs
        self.register_buffer('ema_count', torch.zeros(num_embeddings))
        # same size as dict, initialized as codebook
        # the updated means
        self.register_buffer('ema_weight', torch.empty((self.num_embeddings, self.embedding_dim)))
        self.ema_weight.data.uniform_(-1 / self.num_embeddings, 1 / self.num_embeddings)
        self.decay = decay
        self.epsilon = epsilon

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.IntTensor, float):
        """
        :param x: tensors (output of the Encoder - B,D,H,W).
        :return quantized_x (B, D, H, W), detached codes (B, H*W), latent_loss
        """

        # b, c, h, w = x.shape
        b, t, d = x.shape
        device = x.device
        # Flat input to vectors of embedding dim = C.
        # flat_x = rearrange(x, 'b c h w -> (b h w) c')
        flat_x = rearrange(x, 'b t d -> (b t) d')

        # Calculate distances of each vector w.r.t the dict
        # distances is a matrix (B*H*W, codebook_size)
        distances = (torch.sum(flat_x ** 2, dim=1, keepdim=True)
                     + torch.sum(self.codebook.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_x, self.codebook.weight.t()))

        # Get indices of the closest vector in dict, and create a mask on the correct indexes
        # encoding_indices = (num_vectors_in_batch, 1)
        # Mask = (num_vectors_in_batch, codebook_dim)
        encoding_indices = torch.argmin(distances, dim=1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=device)
        encodings.scatter_(1, encoding_indices.unsqueeze(1), 1)

        # Quantize and un-flat
        quantized = torch.matmul(encodings, self.codebook.weight)

        # Use EMA to update the embedding vectors
        # Update a codebook vector as the mean of the encoder outputs that are closer to it
        # Calculate the usage count of codes and the mean code, then update the codebook vector dividing the two
        if self.training:
            with torch.no_grad():
                ema_count = self.get_buffer('ema_count') * self.decay + (1 - self.decay) * torch.sum(encodings, 0)

                # Laplace smoothing of the ema count
                self.ema_count = (ema_count + self.epsilon) / (b + self.num_embeddings * self.epsilon) * b

                dw = torch.matmul(encodings.t(), flat_x)
                self.ema_weight = self.get_buffer('ema_weight') * self.decay + (1 - self.decay) * dw

                self.codebook.weight.data = self.get_buffer('ema_weight') / self.get_buffer('ema_count').unsqueeze(1)

        # Loss function (only the inputs are updated)
        e_loss = self.commitment_cost * F.mse_loss(quantized.detach(), flat_x)

        # during backpropagation quantized = inputs (copy gradient trick)
        quantized = flat_x + (quantized - flat_x).detach()

        # quantized = rearrange(quantized, '(b h w) c -> b c h w', b=b, h=h, w=w)
        quantized = rearrange(quantized, '(b t) d -> b t d', b=b, t=t)
        # encoding_indices = rearrange(encoding_indices, '(b h w)-> b (h w)', b=b, h=h, w=w).detach()
        encoding_indices = rearrange(encoding_indices, '(b t) -> b t', b=b, t=t).detach()
        
        e_mean = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))
        # return quantized, encoding_indices, e_loss
        return quantized, e_loss, (perplexity,encodings,encoding_indices)

    @torch.no_grad()
    def vec_to_codes(self, x: torch.Tensor) -> torch.IntTensor:
        """
        :param x: tensors (output of the Encoder - B,D,H,W).
        :return flat codebook indices (B, H * W)
        """
        # b, c, h, w = x.shape
        b, t, d = x.shape
        # Flat input to vectors of embedding dim = C.
        # flat_x = rearrange(x, 'b c h w -> (b h w) c')
        flat_x = rearrange(x, 'b t d -> (b t) d')

        # Calculate distances of each vector w.r.t the dict
        # distances is a matrix (B*H*W, codebook_size)
        distances = (torch.sum(flat_x ** 2, dim=1, keepdim=True)
                     + torch.sum(self.codebook.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_x, self.codebook.weight.t()))

        # Get indices of the closest vector in dict
        encoding_indices = torch.argmin(distances, dim=1)
        # encoding_indices = rearrange(encoding_indices, '(b h w) -> b (h w)', b=b, h=h, w=w)
        encoding_indices = rearrange(encoding_indices, '(b t) -> b t', b=b, t=t)

        return encoding_indices
    
    def get_distance(self, x):
        b, t, d = x.shape
        device = x.device
        flat_x = rearrange(x, 'b t d -> (b t) d')
        d = (torch.sum(flat_x ** 2, dim=1, keepdim=True)
                     + torch.sum(self.codebook.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_x, self.codebook.weight.t()))
        d = torch.reshape(d, (x.shape[0], -1, x.shape[2])).permute(0,2,1).contiguous()
        return d
    
    def get_codebook_entry(self, indices, shape):
        """
        indices: (any number of indices,)
        """
        min_encodings = torch.zeros(indices.shape[0], self.num_embeddings).to(indices) # (BS*latent_t, n_e)
        min_encodings.scatter_(1, indices[:,None], 1) # (BS*latent_t, n_e)

        # get quantized latent vectors
        #print(min_encodings.shape, self.embedding.weight.shape)
        z_q = torch.matmul(min_encodings.float(), self.codebook.weight)

        if shape is not None:
            z_q = z_q.view(shape)
            # reshape back to match original input shape
            #z_q = z_q.permute(0, 3, 1, 2).contiguous()
        return z_q
    
