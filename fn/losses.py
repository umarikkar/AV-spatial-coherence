import torch
import torch.nn as nn
from einops import rearrange
import matplotlib.pyplot as plt

import os


import core.config as conf

def plot_scores(scores_raw, scores_all, save=False):

    sc_raw = scores_raw.cpu().detach().numpy()
    sc_all = scores_all.cpu().detach().numpy()

    plt.figure(
        figsize=(8,4)
    )
    plt.subplot(1,2,1)
    plt.imshow(sc_raw)
    plt.title('(a)')
    # plt.axis('off')
    plt.xlabel('image -- image flipped')
    plt.ylabel('audio flipped -- audio')
    plt.colorbar()
    plt.subplot(1,2,2)
    plt.imshow(sc_all)
    plt.colorbar()
    plt.title('(b)')
    # plt.axis('off')
    if save:
        plt.savefig(os.path.join(conf.filenames['net_folder_path'], conf.filenames['net_name'] + '.pdf' ))
    plt.show()

    return
    


class loss_VSL(nn.Module):

    def __init__(self, tau=0.1, eps=1e-4, contrast_param=conf.contrast_param):
        super().__init__()
        self.tau=tau
        self.eps=eps
        self.flip_neg = contrast_param['flip_img'] or contrast_param['flip_mic']
        self.hard_neg = contrast_param['hard_negatives']

    def get_score_semantic(self, scores_raw, neg_mask):

        scores_all = torch.exp(scores_raw/self.tau)*neg_mask

        s_pos = scores_all.diag()
        s_neg_a = scores_all.sum(dim=1)
        s_neg_v = scores_all.sum(dim=0)

        L_av = -1 * torch.log10((s_pos + self.eps) / (s_neg_a + self.eps))
        L_va = -1 * torch.log10((s_pos + self.eps) / (s_neg_v + self.eps))

        L = (0.5*L_av + 0.5*L_va).sum() / (neg_mask.sum() + self.eps)

        return L

    def get_score_spatial(self, scores_raw, beta=None, device=conf.training_param['device']):


        bs = scores_raw.shape[0] // 2

        eye = torch.eye(bs)

        neg_mask = torch.cat((torch.cat((eye, eye), dim=0), torch.cat((eye, eye), dim=0)), dim=1).to(device)
            

        scores_all = torch.exp(scores_raw/self.tau)*neg_mask


        if beta is None:
            beta = bs

        scores_pos1 = scores_all[:bs, :bs]
        scores_pos2 = scores_all[bs:, bs:]

        scores_neg1 = scores_all[:bs, bs:]
        scores_neg2 = scores_all[bs:, :bs]

        s_pos = scores_pos1.diag() + scores_pos2.diag()
        s_neg = scores_neg1.diag() + scores_neg2.diag()

        loss = -1 * torch.log10((s_pos + self.eps) / (s_pos + beta*s_neg + self.eps))

        L = loss.sum() / (neg_mask.sum() + self.eps)

        # L = 0.5*loss_1 + 0.5*loss_2

        return L

    def get_loss(self, scores_raw, neg_mask):

        if not self.hard_neg and self.flip_neg: # keyword-mix
            loss_semantic = self.get_score_semantic(scores_raw, neg_mask)
            loss_spatial = self.get_score_spatial(scores_raw)
            loss_total = 0.5*loss_spatial + 0.5*loss_semantic

        elif self.hard_neg and self.flip_neg: # keyword-hrd
            loss_semantic = None
            loss_spatial = self.get_score_spatial(scores_raw)
            loss_total = loss_spatial

        else: # keyword-none (normal setting-only semantic)
            loss_semantic = self.get_score_semantic(scores_raw, neg_mask)
            loss_spatial = None
            loss_total = loss_semantic

        loss = {
                'total': loss_total,
                'easy': loss_semantic,
                'hard': loss_spatial,
            }

        return loss

    def forward(self, scores_raw, neg_mask):

        bs = scores_raw.shape[0]

        loss = self.get_loss(scores_raw, neg_mask)

        zer = torch.zeros((bs,bs))
        one = torch.ones((bs, bs))
        ide = torch.eye(bs)

        labels = -1*one + 2*ide
        # labels = torch.cat((torch.cat((-1*one + 2*ide, -1*ide), dim=0), torch.cat((-1*ide, ide), dim=0)), dim=1)

        loss_perfect = self.get_loss(labels.to(device=conf.training_param['device']), neg_mask)

        return loss, loss_perfect



class loss_AVOL(nn.Module):

    def __init__(self, eps=1e-8, contrast_param=conf.contrast_param):
        super().__init__()
        self.loss_fn = nn.BCELoss(reduction='none')
        self.eps=eps
        self.flip_neg = contrast_param['flip_img'] or contrast_param['flip_mic']
        self.hard_neg = contrast_param['hard_negatives']
        

    def get_score_semantic(self, scores_raw, neg_mask, device=conf.training_param['device']):

        scores_all = scores_raw * neg_mask

        # plot_scores(scores_raw, scores_all, save=False)

        bs = scores_all.shape[0]

        labels = torch.eye(bs).to(device).reshape((-1,)).unsqueeze(-1)
        scores = scores_all.reshape((-1,)).unsqueeze(-1)

        scores[scores>1], scores[scores<0] = 1, 0

        loss = self.loss_fn(scores, labels).reshape((bs, bs))*neg_mask

        loss_pos = loss.diag().mean()
        loss_neg = (loss.sum() - loss.diag().sum() + self.eps) / ((neg_mask.sum() - bs) + self.eps)

        L = 0.5*loss_pos + 0.5*loss_neg

        return L

    def get_score_spatial(self, scores_raw, _, device=conf.training_param['device']):

        scores_all = scores_raw * 1 

        # plot_scores(scores_all, scores_all)

        bs = scores_all.shape[0] // 2
        s_pos = scores_all.diag()

        s_neg = torch.concat((scores_all[:bs, bs:].diag(), scores_all[bs:, :bs].diag()), dim=0)

        l_pos = torch.ones((2*bs,)).to(device)
        l_neg = torch.zeros((2*bs,)).to(device)

        scores = torch.concat((s_pos, s_neg), dim=0)
        labels = torch.concat((l_pos, l_neg), dim=0)

        L = self.loss_fn(scores, labels).mean()

        return L


    def forward(self, scores_raw, neg_mask, alpha=conf.contrast_param['alpha'], device=conf.training_param['device']):

        if not self.hard_neg and self.flip_neg: # keyword-mix
            loss_semantic = self.get_score_semantic(scores_raw, neg_mask)
            loss_spatial = self.get_score_spatial(scores_raw, neg_mask)
            loss_total = 0.5*loss_spatial + 0.5*loss_semantic

        elif self.hard_neg and self.flip_neg: # keyword-hrd
            loss_semantic = None
            loss_spatial = self.get_score_spatial(scores_raw, neg_mask)
            loss_total = loss_spatial

        else: # keyword-none (normal setting-only semantic)
            loss_semantic = self.get_score_semantic(scores_raw, neg_mask)
            loss_spatial = None
            loss_total = loss_semantic

        loss = {
                'total': loss_total,
                'easy': loss_semantic,
                'hard': loss_spatial,
            }


        loss_perfect = {'total': 0.00,
                        'random': -1*torch.log10(torch.tensor([0.5])) }

        return loss, loss_perfect # perfect loss here is 0.00 (because it is nn.BCE)



class loss_AVE(nn.Module):

    def __init__(self, eps=1e-8):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(reduction='none')
        self.eps=eps

    def forward(self, scores_raw, neg_mask, device=conf.training_param['device'], return_err = False):

        bs = scores_raw.shape[0]

        labels_pos = torch.eye(bs).unsqueeze(-1)
        labels_neg = (-1*torch.eye(bs) + 1).unsqueeze(-1)

        labels = torch.concat((labels_pos, labels_neg), dim=-1).to(device=device)

        if not return_err:

            scores = rearrange(scores_raw, 'a1 a2 c -> (a1 a2) c')
            labels = rearrange(labels, 'a1 a2 c -> (a1 a2) c')

            loss = rearrange(self.loss_fn(scores, labels), '(a1 a2) -> a1 a2', a1=bs)
            loss = loss*neg_mask
            loss = (torch.ones((bs,bs)) + (bs-1)*torch.eye(bs)).to(device=device)*loss

            loss = loss.sum() / (neg_mask.sum() + self.eps)

            loss = {
                'total': loss,
                'easy': None,
                'hard': None,
            }


            loss_perfect = {'total': 0.00,
                            'random': -1*torch.log10(torch.tensor([0.5])) }

            return loss, loss_perfect # perfect loss here is 0.00 (because it is nn.BCE)

        else:

            # bs = scores_raw.shape[0]

            # bs_pos = bs // 2

            # labels_pos = torch.cat((torch.ones((bs_pos, 1)), torch.zeros((bs_pos, 1))), dim=-1)
            # labels_neg = torch.cat((torch.zeros((bs_pos, 1)), torch.ones((bs_pos, 1))), dim=-1)

            # labels = torch.cat((labels_pos, labels_neg), dim=0).to(device)

            _, idx1 = torch.max(scores_raw, dim=-1)
            _, idx2 = torch.max(labels, dim=-1)

            # idx1 = rearrange(idx1, '(a1 a2) -> a1 a2', a1=bs)
            # idx2 = rearrange(idx2, '(a1 a2) -> a1 a2', a1=bs)

            err_idxs = torch.abs(idx1-idx2)*neg_mask

            err_pos = torch.diag(err_idxs)

            n_total = neg_mask.sum()
            n_pos = (neg_mask*(torch.eye(bs).to(device=device))).sum()
            n_neg = n_total - n_pos
            
            e_total = err_idxs.sum()
            e_pos = err_pos.sum()
            e_neg = e_total - e_pos

            errors = {
                'n_total':n_total,
                'n_pos':n_pos,
                'n_neg':n_neg,
                'e_total':e_total,
                'e_pos':e_pos,
                'e_neg':e_neg
            }

            return errors


class loss_embed(nn.Module):

    def __init__(self, eps=1e-8, contrast_param=conf.contrast_param):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()
        self.eps=eps

    def get_score_semantic(self, scores_raw, neg_mask, device=conf.training_param['device']):

        scores = rearrange(scores_raw, 'b h w -> b (h w)')

        hots = torch.argmax(scores, dim=1)

        label = torch.zeros_like(scores)

        for id, num in enumerate(hots):
            label[id, num] = 1.0

        loss = self.loss_fn(scores, label)

        return loss

    def get_score_spatial(self, scores_raw, _, device=conf.training_param['device']):

        return None


    def forward(self, scores_raw, neg_mask, alpha=conf.contrast_param['alpha'], device=conf.training_param['device']):

        loss_semantic = self.get_score_semantic(scores_raw, neg_mask)
        loss_spatial = None
        loss_total = loss_semantic

        loss = {
                'total': loss_total,
                'easy': loss_semantic,
                'hard': loss_spatial,
            }

        loss_perfect = {'total': 0.00,
                        'random': -1*torch.log10(torch.tensor([0.5])) }

        return loss, loss_perfect # perfect loss here is 0.00 (because it is nn.BCE)
