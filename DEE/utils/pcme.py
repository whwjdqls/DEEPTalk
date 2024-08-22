"""Batch-wise efficient probabilistic embedding loss for cross-modal retrieval

Reference code: https://github.com/naver-ai/pcme/blob/main/criterions/probemb.py
"""
import warnings

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def sample_gaussian_tensors(mu, logsigma, num_samples, normalize=False):
    if num_samples == 0:
        return mu.unsqueeze(1)
    logsigma = logsigma * 0.5
    eps = torch.randn(mu.size(0), num_samples, mu.size(1), # BS, num_samples, dim
                      dtype=mu.dtype, device=mu.device)

    samples = eps.mul(torch.exp(logsigma.unsqueeze(1))).add_(
        mu.unsqueeze(1))
    if normalize:
        return F.normalize(samples, p=2, dim=-1)
    else:
        return samples


def batchwise_cdist(samples1, samples2, eps=1e-6, squared=False):
    """Compute L2 distance between each pair of the two multi-head embeddings in batch-wise.
    We may assume that samples have shape N x K x D, N: batch_size, K: number of embeddings, D: dimension of embeddings.
    The size of samples1 and samples2 (`N`) should be either
    - same (each sample-wise distance will be computed separately)
    - len(samples1) = 1 (samples1 will be broadcasted into samples2)
    - len(samples2) = 1 (samples2 will be broadcasted into samples1)

    The following broadcasting operation will be computed:
    (N x 1 x K x D) - (N x K x 1 x D) = (N x K x K x D)

    Parameters
    ----------
    samples1: torch.Tensor (shape: N x K x D)
    samples2: torch.Tensor (shape: N x K x D)

    Returns
    -------
    batchwise distance: N x K ** 2
    """
    if len(samples1.size()) != 3 or len(samples2.size()) != 3:
        raise RuntimeError(
            'expected: 3-dim tensors, got: {}, {}'.format(samples1.size(), samples2.size()))

    if samples1.size(0) == samples2.size(0):
        batch_size = samples1.size(0)
    elif samples1.size(0) == 1:
        batch_size = samples2.size(0)
    elif samples2.size(0) == 1:
        batch_size = samples1.size(0)
    else:
        raise RuntimeError(f'samples1 ({samples1.size()}) and samples2 ({samples2.size()}) dimensionalities '
                           'are non-broadcastable.')

    samples1 = samples1.unsqueeze(1)
    samples2 = samples2.unsqueeze(2)
    if squared:
        return ((samples1 - samples2) ** 2).sum(-1).view(batch_size, -1)
    else:
        return torch.sqrt(((samples1 - samples2) ** 2).sum(-1) + eps).view(batch_size, -1)


def soft_contrastive_nll(logit, matched):
    r"""Compute the negative log-likelihood of the soft contrastive loss.

    .. math::
        NLL_{ij} = -\log p(m = m_{ij} | z_i, z_j)
                 = -\log \left[ \mathbb{I}_{m_{ij} = 1} \sigma(-a \| z_i - z_j \|_2 + b)
                         +  \mathbb{I}_{m_{ij} = -1} (1 - \sigma(-a \| z_i - z_j \|_2 + b)) \right].

    Note that the matching indicator {m_ij} is 1 if i and j are matched otherwise -1.
    Here we define the sigmoid function as the following:
    .. math::
        \sigma(x) = \frac{\exp(x)}{\exp(x) + \exp(-x)}, \text{ i.e., }
        1 - \sigma(x) = \frac{\exp(-x)}{\exp(x) + \exp(-x)}.

    Here we sample "logit", s_{ij} by Monte-Carlo sampling to get the expected soft contrastive loss.
    .. math::
        s_{ij}^k = -a \| z_i^k - z_j^k \|_2 + b, z_i^k ~ \mathcal N (\mu_i, \Sigma_i), z_j^k ~ \mathcal N (\mu_j, \Sigma_j).

    Then we can compute NLL by logsumexp (here, we omit `k` in s_{ij}^k for the simplicity):
    .. math::
        NLL_{ij} = -\log \left[ \frac{1}{K^2} \sum_{s_{ij}} \left{ \frac{\exp(s_{ij} m_ij)}{\exp(s_{ij}) + \exp(-s_{ij})} \right} \right]
                 = (\log K^2) -\log \sum_{s_{ij}} \left[ \exp \left( s_{ij} m_ij - \log(\exp(s_{ij} + (-s_{ij}))) \right) \right]
                 = (\log K^2) -logsumexp( s_{ij} m_{ij} - logsumexp(s_{ij}, -s_{ij}) ).

    Parameters
    ----------
    logit: torch.Tensor (shape: N x K ** 2)
    matched: torch.Tensor (shape: N), an element should be either 1 (matched) or -1 (mismatched)

    Returns
    -------
    NLL loss: torch.Tensor (shape: N), should apply `reduction` operator for the backward operation.
    """
    if len(matched.size()) == 1:
        matched = matched[:, None]
    return -(
        (logit * matched - torch.stack(
            (logit, -logit), dim=2).logsumexp(dim=2, keepdim=False)
         ).logsumexp(dim=1)) + np.log(logit.size(1))


class MCSoftContrastiveLoss(nn.Module):
    r"""Creates a criterion that measures the pairwise soft contrastive loss given
    input tensor pairs :math:`X`, :math:`Y` where each tensor is already sampled from a distribution.

    .. math::
        \log p(m = \hat m | x, y)
        p(m = 1 | x, y) = \sigma(-a \| x - y \|_2 + b)
        p(m = 0 | x, y) = 1 - \sigma(-a \| x - y \|_2 + b)
        \sigma(x) = \frac{\exp(x)}{\exp(x) + \exp(-x)}, \text{ i.e., }
        1 - \sigma(x) = \frac{\exp(-x)}{\exp(x) + \exp(-x)}.

    This code assumes that :math:`x_i` and :math:`y_j` are in same class if i = j,
    and in different class otherwise.

    The division by :math:`n` can be avoided if sets ``reduction = 'sum'``.

    Parameters
    ----------
    TBD

    Shape
    -----
    Input1 : torch.Tensor
        :math:`(N, K, D)` shape, `N` is the batch size, `K` is the number of samples and `D` is the size of a sample.
    Input2: torch.Tensor
        :math:`(N, K, D)` shape, `N` is the batch size, `K` is the number of samples and `D` is the size of a sample.
    Output: torch.Tensor
        If :attr:`reduction` is ``'none'``, then :math:`(N)`.
    """

    def __init__(
            self,
            args,
            init_shift=5,
            init_negative_scale=5,
            num_samples=8,
            vib_beta=0,
            pdist_fn='iterative',
            prob_distance='non_squared_l2',
            **kwargs):
        super().__init__()
        self.add_n_mog = args.add_n_mog

        shift = init_shift * torch.ones(1)
        negative_scale = init_negative_scale * torch.ones(1)

        shift = nn.Parameter(shift)
        negative_scale = nn.Parameter(negative_scale)

        self.register_parameter('shift', shift)
        self.register_parameter('negative_scale', negative_scale)

        self.num_samples = num_samples
        self.vib_beta = vib_beta
        self.prob_distance = prob_distance

        if self.prob_distance == 'non_squared_l2':
            self.squared = False
        elif self.prob_distance == 'squared_l2':
            self.squared = True
        else:
            raise ValueError(
                f'{self.prob_distance=} is not a valid ("non_squared_l2", "squared_l2")')

        if pdist_fn not in {'batchwise_cdist', 'iterative'}:
            raise ValueError(
                f'{pdist_fn=} is not a valid pdist_fn ("batchwise_cdist", "iterative")')
        self.pdist_fn = pdist_fn

    def max_violation_on(self):
        warnings.warn(
            'PCME loss does not support max violation. Nothing happens')
        return

    def max_violation_off(self):
        warnings.warn(
            'PCME loss does not support max violation. Nothing happens')
        return

    def kl_divergence(self, mu, logsigma):
        kl_loss = -0.5 * (1 + logsigma - mu.pow(2) - logsigma.exp()).mean()
        if kl_loss > 10000:
            # XXX prevent loss exploration
            warnings.warn(f'Detected a VIB loss explosion ({kl_loss=} > 10000). Ignore the VIB loss for stability.')
            return 0
        return kl_loss

    def _pdist(self, input1, input2, averaged=False):
        if self.pdist_fn == 'batchwise_cdist':
            candidates = []
            selected_idx = []
            for i in range(len(input1)):
                for j in range(len(input2)):
                    candidates.append(i)
                    selected_idx.append(j)
            anchor_idx = torch.from_numpy(
                np.array(candidates)).long().to(input1.device)
            selected_idx = torch.from_numpy(
                np.array(selected_idx)).long().to(input1.device)

            anchors = input1[anchor_idx]
            selected = input2[selected_idx]

            cdist = batchwise_cdist(anchors, selected, squared=self.squared)
            if averaged:
                return cdist.mean(dim=1)
            else:
                return cdist
        elif self.pdist_fn == 'iterative':
            if averaged:
                dist = 0
            else:
                dist = []
            for idx1 in range(self.num_samples):
                for idx2 in range(self.num_samples):
                    x = input1[:, idx1, :]
                    y = input2[:, idx2, :]
                    _dist = ((x.unsqueeze(1) - y.unsqueeze(0)) ** 2).sum(-1).view(-1)
                    if not self.squared:
                        _dist = _dist.sqrt()
                    if averaged:
                        dist += (_dist / self.num_samples / self.num_samples)
                    else:
                        dist.append(_dist)
            if not averaged:
                dist = torch.stack(dist, dim=1)

            return dist

    def _compute_loss(self, input1, input2, matched):
        cdist = self._pdist(input1, input2)
        logits = -self.negative_scale * cdist + self.shift

        matched = matched.view(-1)
        matched = 2 * matched - 1

        idx = matched == 1
        loss_pos = soft_contrastive_nll(logits[idx], matched[idx]).sum()
        idx = matched != 1

        loss_neg = soft_contrastive_nll(logits[idx], matched[idx]).sum()

        loss = loss_pos + loss_neg
        return {
            'loss': loss,
            'pos_loss': loss_pos,
            'neg_loss': loss_neg,
        }

    def forward(self, img_emb, cap_emb, matched=None, hard_matched=None):
        if self.add_n_mog >=1:
            n_gaussians = self.add_n_mog+1
            sampled_image_features = []
            sampled_caption_features = []
            for i in range(n_gaussians):
                sampled_image_features.append(sample_gaussian_tensors(# BS, num_samples, dim
                    img_emb['mean'][:,i,:], img_emb['std'][:,i,:], self.num_samples))
                sampled_caption_features.append(sample_gaussian_tensors(
                    cap_emb['mean'][:,i,:], cap_emb['std'][:,i,:], self.num_samples))
            sampled_image_features = torch.concat(sampled_image_features, dim=1) # (Bs, n_gaussians*num_samples, dim)
            sampled_caption_features = torch.concat(sampled_caption_features, dim=1)
        else:
            sampled_image_features = sample_gaussian_tensors(
                img_emb['mean'], img_emb.get('std'), self.num_samples)
            sampled_caption_features = sample_gaussian_tensors(
                cap_emb['mean'], cap_emb.get('std'), self.num_samples)
        vib_loss, vib_loss_val = 0, 0

        if self.vib_beta != 0:
            vib_loss =\
                self.kl_divergence(img_emb['mean'], img_emb['std']) + \
                self.kl_divergence(cap_emb['mean'], cap_emb['std'])
            vib_loss_val = vib_loss
        

        if matched is None:
            matched = torch.eye(len(img_emb['mean'])).to(img_emb['mean'].device)

        loss = self._compute_loss(sampled_image_features, sampled_caption_features, matched=matched)
        # NOTE: Efficient implementation for
        # when i2t loss and t2i loss are the same (https://github.com/naver-ai/pcme/issues/3)
        loss = 2 * loss['loss'] + self.vib_beta * vib_loss

        loss_dict = {
            'loss/audio_std': cap_emb['std'].mean().item(),
            'loss/exp_std': img_emb['std'].mean().item(),
            'loss/audio_mean' : cap_emb['mean'].mean().item(),
            'loss/exp_mean' : img_emb['mean'].mean().item(),
            'loss/loss': loss.item(),
            'criterion/shift': self.shift.item(),
            'criterion/negative_scale': self.negative_scale.item(),
        }

        if self.vib_beta != 0:
            loss_dict['loss/vib_loss'] = vib_loss_val

        return loss, loss_dict


