import torch
from torch import nn
import torch.nn.functional as F


def CLIP_loss(audio_embedding, expression_embedding, temperature, device):
    labels = torch.arange(audio_embedding.shape[0], device=device, dtype=torch.long) # (batch_size)
    # add clipping
    logits = expression_embedding @ audio_embedding.T * torch.clip(temperature.exp(), max = 100) # (batch_size, batch_size)
    loss_expression =  F.cross_entropy(logits, labels) 
    loss_audio =  F.cross_entropy(logits.T, labels.T)
    loss = (loss_expression + loss_audio) / 2.
    return loss

def CLIP_loss_with_expression_guide(audio_embedding, expression_embedding ,expression_guide_weight, temperature, device):
    '''
    added expression guided audio loss along with the CLIP loss
    '''
    labels = torch.arange(audio_embedding.shape[0], device=device, dtype=torch.long) # (batch_size)
    # add clipping
    logits = expression_embedding @ audio_embedding.T * torch.clip(temperature.exp(), max = 100) # (batch_size, batch_size)
    loss_expression =  F.cross_entropy(logits, labels) 
    loss_audio =  F.cross_entropy(logits.T, labels.T)
    
    # added so that the audio_embedding will come closer to expression_embedding, using expresison embedding as a guide
    cosine_similarity = F.cosine_similarity(expression_embedding.detach(), audio_embedding, dim=1)
    cosine_loss = cosine_similarity.mean()  
    loss = (loss_expression + loss_audio) / 2. + ((1-cosine_loss) * expression_guide_weight)
    # loss = (loss_expression + loss_audio) / 2. - (cosine_loss * expression_guide_weight)
    return loss
    
def RECO_loss(audio_embedding, expression_embedding, temperature, device):
    logits = expression_embedding @ audio_embedding.T 
    
    loss_pos = torch.diagonal(logits).add(-1).pow(2).sum() 
    
    off_diag = logits * (1-torch.eye(logits.shape[0], device=device))
    stacked_with_zeros = torch.stack((off_diag, torch.zeros_like(off_diag, device=device)), dim=2) # (BS,BS,2)
    max_elements = torch.max(stacked_with_zeros, dim=2)[0]
    loss_neg = max_elements.pow(2).sum()
    
    loss = loss_pos + 0.6 * loss_neg
    return loss

def emotion_guided_loss_gt(audio_embedding, expression_embedding, emotion,beta, temperature,device):
    '''
    imlementation of emotion guided loss from emotionCLIP paper with ground truth emotion
    beta = 100
    '''
    
    labels = torch.arange(audio_embedding.shape[0], device=device, dtype=torch.long) # (batch_size)
    # add clipping
    logits = expression_embedding @ audio_embedding.T * torch.clip(temperature.exp(), max = 100) # (batch_size, batch_size)
    sentiment_weights = create_label_matrix(emotion).to(device) # (batch_size, batch_size)
    weighted_logits = logits - (beta * sentiment_weights)
    loss_expression =  F.cross_entropy(weighted_logits, labels) 
    loss_audio =  F.cross_entropy(weighted_logits.T, labels.T)
    loss = (loss_expression + loss_audio) / 2.
    return loss
    

def emotion_guided_loss():
    '''
    imlementation of emotion guided loss from emotionCLIP paper
    '''
    pass

def actor_guided_loss():
    '''
    peanalize more if the actor is same
    '''
    pass


def create_label_matrix(labels):
    matrix_size = len(labels)
    label_matrix = torch.ones((matrix_size, matrix_size)) 
    for i in range(matrix_size):
        for j in range(matrix_size):
            if i != j: # for non diagonal elements
                if labels[i] != labels[j]: # if the labels are different
                    label_matrix[i][j] = 0 # 
    label_matrix.fill_diagonal_(0)
    return label_matrix

def create_matching_matrix(labels):
    # this function is for using emotion supervision in prob DEE
    matrix_size = len(labels)
    label_matrix = torch.zeros((matrix_size, matrix_size)) 
    for i in range(matrix_size):
        for j in range(matrix_size):
            if labels[i] == labels[j]: # if the labels are different
                label_matrix[i][j] = 1 # 
    return label_matrix

""" Improved probabilistic embedding loss for cross-modal retrieval

PCME++
Copyright (c) 2023-present NAVER Cloud Corp.
MIT license
"""
import warnings

class ClosedFormSampledDistanceLoss(nn.Module):
    def __init__(
            self,
            init_shift=5, # b initialized to 5
            init_negative_scale=5,# a initialized to 5
            vib_beta=0, # beta for VIB loss
            smoothness_alpha=0, # alpha for Pseudo-Positive Pair
            prob_distance='csd',
            **kwargs):
        super().__init__()

        shift = init_shift * torch.ones(1) # b in original paper
        negative_scale = init_negative_scale * torch.ones(1)# a in original paper

        shift = nn.Parameter(shift)
        negative_scale = nn.Parameter(negative_scale)

        self.register_parameter('shift', shift)
        self.register_parameter('negative_scale', negative_scale)

        self.vib_beta = vib_beta
        self.smoothness_alpha = smoothness_alpha

        # XXX Do not specify prob_distance unless for the prob dist ablation study
        self.prob_distance = prob_distance 

        self.bceloss = nn.BCEWithLogitsLoss()

        if self.prob_distance not in {'csd', 'wdist'}:
            raise ValueError(f'Invalid prob_distance. Expected ("csd", "wdist"), but {prob_distance=}')

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

    def _recompute_matched(self, matched, logits, smoothness=0):
        """ Recompute the `matched` matrix if the smoothness value is given.
        """
        if not smoothness:
            return matched, None
        else:
            logits = logits.view(matched.size())
            # XXX Warning: all negative pairs will return weird results
            gt_labels, gt_indices = torch.max(matched, dim=1) # get pair indices
            gt_vals = logits[:, gt_indices].diag() # get the logits of the pairs
            pseudo_gt_indices = (logits >= gt_vals.unsqueeze(1)) # if the logits are greater than the pair logits, then it is a pseudo pair
            new_matched = (gt_labels.unsqueeze(1) * (pseudo_gt_indices))
            _matched = matched.clone()
            _matched[pseudo_gt_indices] = new_matched[pseudo_gt_indices]

            return _matched, torch.sum(pseudo_gt_indices).item() - len(gt_indices)

    def _compute_prob_matching_loss(self, logits, matched, smoothness=0):
        matched, n_pseudo_gts = self._recompute_matched(matched, logits, smoothness) 
        loss = self.bceloss(logits, matched)

        return {
            'loss': loss,
            'n_pseudo_gts': n_pseudo_gts,
        }

    def _compute_closed_form_loss(self, input1, input2, matched, smoothness=0):
        """ Closed-form probabilistic matching loss -- See Eq (1) and (2) in the paper.
        """
        mu_pdist = ((input1['mean'].unsqueeze(1) - input2['mean'].unsqueeze(0)) ** 2).sum(-1)
        sigma_pdist = ((torch.exp(input1['std']).unsqueeze(1) + torch.exp(input2['std']).unsqueeze(0))).sum(-1)
        logits = mu_pdist + sigma_pdist
        logits = -self.negative_scale * logits + self.shift # (Bs, BS)
        loss_dict = self._compute_prob_matching_loss(logits, matched, smoothness=smoothness)
        loss_dict['loss/mu_pdist'] = mu_pdist.mean()
        loss_dict['loss/sigma_pdist'] = sigma_pdist.mean()
        return loss_dict

    def _compute_wd_loss(self, input1, input2, matched, smoothness=0):
        """ Wasserstien loss (only used for the ablation study)
        """
        mu_pdist = ((input1['mean'].unsqueeze(1) - input2['mean'].unsqueeze(0)) ** 2).sum(-1).view(-1)
        sigma_pdist = ((torch.exp(input1['std'] / 2).unsqueeze(1) - torch.exp(input2['std'] / 2).unsqueeze(0)) ** 2).sum(-1).view(-1)

        logits = mu_pdist + sigma_pdist
        logits = logits.reshape(len(input1['mean']), len(input2['mean']))
        logits = -self.negative_scale * logits + self.shift
        loss_dict = self._compute_prob_matching_loss(logits, matched, smoothness=smoothness)
        loss_dict['loss/mu_pdist'] = mu_pdist.mean()
        loss_dict['loss/sigma_pdist'] = sigma_pdist.mean()
        return loss_dict

    def forward(self, img_emb, cap_emb, matched=None):
        if self.prob_distance == 'wdist':
            loss_fn = self._compute_wd_loss
        else:
            loss_fn = self._compute_closed_form_loss # we are using
        vib_loss = 0

        if self.vib_beta != 0:
            vib_loss =\
                self.kl_divergence(img_emb['mean'], img_emb['std']) + \
                self.kl_divergence(cap_emb['mean'], cap_emb['std'])

        if matched is None:
            matched = torch.eye(len(img_emb['mean'])).to(img_emb['mean'].device)

        loss = loss_fn(img_emb, cap_emb, matched=matched) # no smoothness here
        # NOTE: Efficient implementation for
        # when i2t loss and t2i loss are the same (https://github.com/naver-ai/pcme/issues/3)
        loss = 2 * loss['loss'] + self.vib_beta * vib_loss
        # loss dict has ['loss', 'n_pseudo_gts', 'loss/mu_pdist', 'loss/sigma_pdist']
        # loss_dict = {
        #     'loss/loss': loss,
        #     'criterion/shift': self.shift,
        #     'criterion/negative_scale': self.negative_scale,
        # }
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
            loss_dict['loss/vib_loss'] = vib_loss

        if self.smoothness_alpha:
            smooth_i2t_loss = loss_fn(img_emb, cap_emb, matched=matched, smoothness=self.smoothness_alpha)
            smooth_t2i_loss = loss_fn(cap_emb, img_emb, matched=matched.T, smoothness=self.smoothness_alpha)
            loss = loss + self.smoothness_alpha * (smooth_i2t_loss['loss'] + smooth_t2i_loss['loss'])
            loss_dict['loss/loss'] = loss
            loss_dict['loss/n_pseudo_gts'] = smooth_i2t_loss['n_pseudo_gts'] + smooth_t2i_loss['n_pseudo_gts']

        return loss, loss_dict
