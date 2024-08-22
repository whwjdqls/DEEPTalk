"""
PCME++
Copyright (c) 2023-present NAVER Cloud Corp.
MIT license
"""
import os
import warnings

import numpy as np
import torch

# from eccv_caption import Metrics


def compute_matmul_sims(images, captions, to_double=False):
    if to_double:
        images = images.astype(np.float64)
        captions = captions.astype(np.float64)
    similarities = np.matmul(images, np.matrix.transpose(captions))
    return similarities


def compute_csd_sims(images, captions, image_sigmas, caption_sigmas, to_double=False):
    """ Compute similarity scores while preventing OOM
    """
    if any([torch.is_tensor(x) for x in (images, captions, image_sigmas, caption_sigmas)]):
        raise ValueError('All inputs must be numpy arrays')
    
    csd = 2 - 2 * compute_matmul_sims(images, captions, to_double=to_double)
    image_sigmas = np.exp(image_sigmas)
    caption_sigmas = np.exp(caption_sigmas)
    if to_double:
        image_sigmas = image_sigmas.astype(np.float64)
        caption_sigmas = caption_sigmas.astype(np.float64)
    csd = csd + image_sigmas.sum(-1).reshape((-1, 1))
    csd = csd + caption_sigmas.sum(-1).reshape((1, -1))
    return -csd # image @ caption.T



def compute_matching_prob_sims(images, captions, num_samples, negative_scale, shift):
    # https://github.com/Lightning-AI/metrics/blob/54a06013cdac4895bf8e85b583c5f220388ebc1d/src/torchmetrics/functional/pairwise/euclidean.py#L23
    # XXX upcast to float64? (x.to(torch.float64))
    sims = None
    for img_idx in range(num_samples):
        for cap_idx in range(num_samples):
            x = images[:, img_idx, :]
            y = captions[:, cap_idx, :]
            x_norm = (x * x).sum(dim=1, keepdim=True)
            y_norm = (y * y).sum(dim=1)
            _sims = x_norm + y_norm - 2 * x.mm(y.T)
            _sims = _sims.sqrt()
            _sims = -negative_scale.to(dtype=_sims.dtype, device=_sims.device) * \
                _sims + shift.to(dtype=_sims.dtype, device=_sims.device)
            a = torch.exp(_sims)
            b = torch.exp(-_sims)
            _sims = a / (a + b)
            _sims_np = _sims.detach().cpu().numpy()
            del x, y, x_norm, y_norm, a, b, _sims
            if sims is None:
                sims = _sims_np
            else:
                sims += _sims_np
    return sims


def eval_uncertainty(
        sims, img_sigmas, cap_sigmas, all_iids, all_cids,
        metrics=['eccv_map_at_r', 'eccv_r1', 'cxc_r1', 'coco_5k_r1'],
        n_bins=10):
    uncertainties = {
        'img': np.exp(img_sigmas[::5, :]).sum(axis=1),
        'cap': np.exp(cap_sigmas).sum(axis=1),
    }
    mod_sims = {
        'img': sims,
        'cap': sims.T,
    }
    item_ids = {
        'img': all_iids,
        'cap': all_cids,
    }

    unc_vs_scores = {m: {'img': [], 'cap': []} for m in metrics}

    for mod in ('img', 'cap'):
        unc = uncertainties[mod]
        _sims = mod_sims[mod]
        inds = np.argsort(unc)
        bin_size = len(inds) // n_bins

        if mod == 'img':
            gids = all_cids
            modality = 'i2t'
        else:
            gids = all_iids
            modality = 't2i'

        for hist_idx in range(n_bins):
            cur_inds = inds[range(hist_idx * bin_size,
                                  (hist_idx + 1) * bin_size)]
            cur_scores = eval_mod(
                _sims[cur_inds], item_ids[mod][cur_inds], gids, mod)
            for m in metrics:
                unc_vs_scores[m][mod].append(cur_scores[m][modality])

    return unc_vs_scores


def eval_i2t_val(npts, sims, return_ranks=False, mode='coco'):
    """
    Images->Text (Image Annotation)
    Images: (N, n_region, d) matrix of images
    Captions: (5N, max_n_word, d) matrix of captions
    CapLens: (5N) array of caption lengths
    sims: (N, 5N) matrix of similarity im-cap
    """
    ranks = np.zeros(npts)
    top1 = np.zeros(npts)
    for index in range(npts):
        inds = np.argsort(sims[index])[::-1]
        if mode == 'coco':
            rank = 1e20
            for i in range(5 * index, 5 * index + 5, 1):
                tmp = np.where(inds == i)[0][0]
                if tmp < rank:
                    rank = tmp
            ranks[index] = rank
            top1[index] = inds[0]
        else:
            rank = np.where(inds == index)[0][0]
            ranks[index] = rank
            top1[index] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1

    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)


def eval_t2i_val(npts, sims, return_ranks=False, mode='coco'):
    """
    Text->Images (Image Search)
    Images: (N, n_region, d) matrix of images
    Captions: (5N, max_n_word, d) matrix of captions
    CapLens: (5N) array of caption lengths
    sims: (N, 5N) matrix of similarity im-cap
    """
    # npts = images.shape[0]

    if mode == 'coco':
        ranks = np.zeros(5 * npts)
        top1 = np.zeros(5 * npts)
    else:
        ranks = np.zeros(npts)
        top1 = np.zeros(npts)

    # --> (5N(caption), N(image))
    sims = sims.T

    for index in range(npts):
        if mode == 'coco':
            for i in range(5):
                inds = np.argsort(sims[5 * index + i])[::-1]
                ranks[5 * index + i] = np.where(inds == index)[0][0]
                top1[5 * index + i] = inds[0]
        else:
            inds = np.argsort(sims[index])[::-1]
            ranks[index] = np.where(inds == index)[0][0]
            top1[index] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)


def eval_coco_test_loader(sims, data_path):
    target_metrics = (
        'eccv_r1', 'eccv_map_at_r', 'eccv_rprecision',
        'coco_1k_recalls', 'coco_5k_recalls', 'cxc_recalls')
    Ks = (1, 5, 10)
    scores = eval_all(sims, data_path, target_metrics=target_metrics, Ks=Ks)

    scores['coco_1k'] = {'rsum': 0}
    for modality in ('i2t', 't2i'):
        for K in (1, 5, 10):
            scores['coco_1k']['rsum'] += scores[f'coco_1k_r{K}'][modality]

    report_dict = {}
    for metric_name, metric_dict in scores.items():
        avg_value = 0
        for modality, value in metric_dict.items():
            report_dict[f'eval_{modality}/{metric_name}'] = value * 100
            avg_value += (value * 100)
        avg_value /= len(metric_dict)
        report_dict[f'eval_avg/{metric_name}'] = avg_value
    return report_dict


def eval_coco_uncertainty(sims, img_sigmas, cap_sigmas, data_path, n_bins=10):
    cxc_annot_base = os.path.join(data_path, 'cxc_annots')
    img_id_path = os.path.join(cxc_annot_base, 'testall_ids.txt')
    cap_id_path = os.path.join(cxc_annot_base, 'testall_capids.txt')

    with open(img_id_path) as f:
        all_iids = f.readlines()
        all_iids = all_iids[::5]
    with open(cap_id_path) as f:
        all_cids = f.readlines()
    all_iids = np.array([int(_id) for _id in all_iids])
    all_cids = np.array([int(_id) for _id in all_cids])

    unc_vs_scores = eval_uncertainty(
        sims, img_sigmas, cap_sigmas, all_iids, all_cids,
        metrics=['eccv_map_at_r', 'eccv_r1', 'cxc_r1', 'coco_5k_r1'],
        n_bins=n_bins)

    report_dict = {}
    for metric in ('eccv_map_at_r', 'coco_5k_r1'):
        for mod in ('img', 'cap'):
            report_dict[f'unc_linear/{metric}_{mod}'] = np.corrcoef(
                np.arange(n_bins), unc_vs_scores[metric][mod])[0, 1]
    return report_dict, unc_vs_scores

