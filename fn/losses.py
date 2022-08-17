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

    def __init__(self, tau=0.8, eps=1e-4, alpha=0.5, beta=0.5):
        super().__init__()
        self.tau=tau
        self.eps=eps
        self.alpha=alpha
        self.beta=beta

    def forward(self, scores_raw, neg_mask):

        scores = torch.exp(scores_raw/self.tau)*neg_mask

        s_pos = scores.diag()
        s_neg_a = scores.sum(dim=1)
        s_neg_v = scores.sum(dim=0)

        L_av = -1 * torch.log10((s_pos + self.eps) / (s_neg_a + self.eps))
        L_va = -1 * torch.log10((s_pos + self.eps) / (s_neg_v + self.eps))

        loss = (self.alpha*L_av + self.beta*L_va).sum() / (neg_mask.sum() + self.eps)

        return loss

class loss_VSL_flip(nn.Module):

    def __init__(self, tau=0.5, eps=1e-4, alpha=conf.contrast_param['alpha']):
        super().__init__()
        self.tau=tau
        self.eps=eps
        self.alpha=alpha

    def get_score_spatial(self, scores_raw, neg_mask, alpha=1, beta=None):

        scores_all = torch.exp(scores_raw/self.tau)*neg_mask

        # plot_scores(scores_raw, scores_all)

        bs = scores_all.shape[0] // 2

        if beta is None:
            beta = bs

        scores_pos1 = scores_all[:bs, :bs]
        scores_pos2 = scores_all[bs:, bs:]

        scores_neg1 = scores_all[:bs, bs:]
        scores_neg2 = scores_all[bs:, :bs]

        s_pos1 = scores_pos1.diag()
        s_pos2 = scores_pos2.diag()

        s_neg = scores_neg1.diag() + scores_neg2.diag()

        loss_1 = -1 * torch.log10((s_pos1 + self.eps) / (s_pos1 + 0.5*beta*s_neg + self.eps)).mean()
        loss_2 = -1 * torch.log10((s_pos2 + self.eps) / (s_pos2 + 0.5*beta*s_neg + self.eps)).mean()

        L = 0.5*loss_1 + 0.5*loss_2

        loss = {
            'total': L,
            'easy': None,
            'hard': None,
        }

        return loss

    def get_score_semantic(self, scores_raw, neg_mask, alpha=1, beta=None):

        scores_all = torch.exp(scores_raw/self.tau)*neg_mask

        # plot_scores(scores_raw, scores_all)

        bs = scores_all.shape[0] // 2

        if beta is None:
            beta = bs

        scores_pos1 = scores_all[:bs, :bs]
        scores_pos2 = scores_all[bs:, bs:]

        scores_neg1 = scores_all[:bs, bs:]
        scores_neg2 = scores_all[bs:, :bs]

        s_pos1 = scores_pos1.diag()
        s_pos2 = scores_pos2.diag()

        s_neg = scores_neg1.diag() + scores_neg2.diag()

        loss_1 = -1 * torch.log10((s_pos1 + self.eps) / (s_pos1 + 0.5*beta*s_neg + self.eps)).mean()
        loss_2 = -1 * torch.log10((s_pos2 + self.eps) / (s_pos2 + 0.5*beta*s_neg + self.eps)).mean()

        L = 0.5*loss_1 + 0.5*loss_2

        loss = {
            'total': L,
            'easy': None,
            'hard': None,
        }

        return loss


    def forward(self, scores_raw, neg_mask, alpha=None):

        if alpha is None:
            alpha = self.alpha 

        bs = scores_raw.shape[0] // 2

        loss = self.get_score_spatial(scores_raw, neg_mask, alpha)

        zer = torch.zeros((bs,bs))
        one = torch.ones((bs, bs))
        ide = torch.eye(bs)

        labels = torch.cat((torch.cat((-1*one + 2*ide, -1*ide), dim=0), torch.cat((-1*ide, ide), dim=0)), dim=1)
        
        loss_perfect = self.get_score_spatial(labels.to(device=conf.training_param['device']), neg_mask, alpha)

        return loss, loss_perfect


# class loss_VSL_flip(nn.Module):

#     def __init__(self, tau=0.2, eps=1e-4, alpha=conf.contrast_param['alpha']):
#         super().__init__()
#         self.tau=tau
#         self.eps=eps
#         self.alpha=alpha

#     def get_score(self, scores_raw, neg_mask, alpha=None):


#         scores_all = torch.exp(scores_raw/self.tau)*neg_mask

#         sc_raw = scores_raw.cpu().detach().numpy()
#         sc_all = scores_all.cpu().detach().numpy()
#         plt.figure()
#         plt.subplot(1,2,1)
#         plt.imshow(sc_raw)
#         plt.subplot(1,2,2)
#         plt.imshow(sc_all)
#         plt.show()

#         bs = scores_all.shape[0] // 2

#         scores_easy = scores_all[:bs, :bs]
#         scores_hard = scores_all[bs:, bs:]

#         s_pos = scores_easy.diag()
#         s_hard = scores_hard.diag() + s_pos
#         s_neg_a = scores_easy.sum(dim=1)
#         s_neg_v = scores_easy.sum(dim=0) 
        
#         L_av = (-1 * torch.log10((s_pos + self.eps) / (s_neg_a + self.eps))) . sum()
#         L_va = (-1 * torch.log10((s_pos + self.eps) / (s_neg_v + self.eps))) . sum()
#         L_hd = (-1 * torch.log10((s_pos + self.eps) / (bs*s_hard + self.eps))) . sum()

#         # norm_sum = self.alpha*neg_mask[:bs, :bs].sum() + (1-self.alpha)*bs*bs

#         ls_easy = (L_va+L_av) / (neg_mask[:bs, :bs].sum() + self.eps)
#         ls_hard = L_hd/(2*bs + self.eps)

#         L = (1-alpha)*ls_easy + alpha*ls_hard

#         loss = {
#             'total': L,
#             'easy': ls_easy,
#             'hard': ls_hard,
#         }

#         return loss



#     def forward(self, scores_raw, neg_mask, alpha=None):

#         if alpha is None:
#             alpha = self.alpha 

#         bs = scores_raw.shape[0] // 2

#         loss = self.get_score(scores_raw, neg_mask, alpha)

#         zer = torch.zeros((bs,bs))
#         one = torch.ones((bs, bs))
#         ide = torch.eye(bs)

#         labels = torch.cat((torch.cat((-1*one + 2*ide, zer), dim=0), torch.cat((zer, -1*ide), dim=0)), dim=1)
        
#         loss_perfect = self.get_score(labels.to(device=conf.training_param['device']), neg_mask, alpha)

#         return loss, loss_perfect


class loss_AVOL(nn.Module):

    def __init__(self, eps=1e-8, contrast_param=conf.contrast_param):
        super().__init__()
        self.loss_fn = nn.BCELoss(reduction='none')
        self.eps=eps

    def get_score(self, scores_raw, neg_mask, device=conf.training_param['device']):

        bs = scores_raw.shape[0]

        labels = torch.eye(len(scores_raw)).to(device=device).view(-1,1)
        scores = scores_raw.view(-1,1)

        loss = self.loss_fn(scores, labels).view(bs,bs)
        loss = loss*neg_mask

        loss *= (torch.ones((bs,bs)) + (bs-1)*torch.eye(bs)).to(device=device)
        loss = loss.sum() / (neg_mask.sum() + bs*(bs-1) + self.eps)

        loss = {
            'total': loss,
        }

        return loss


    def forward(self, scores_raw, neg_mask, device=conf.training_param['device'], return_err=False):

        loss = self.get_score(scores_raw, neg_mask)

        return loss, 0.00 # perfect loss here is 0.00 (because it is nn.BCE)



class loss_AVOL_flip(nn.Module):

    def __init__(self, eps=1e-8, contrast_param=conf.contrast_param):
        super().__init__()
        self.loss_fn = nn.BCELoss(reduction='none')
        self.eps=eps
        self.flip_neg = contrast_param['flip_img'] or contrast_param['flip_mic']

    def get_score_semantic(self, scores_raw, neg_mask, device=conf.training_param['device']):

        scores_all = scores_raw * neg_mask

        plot_scores(scores_raw, scores_all, save=False)

        bs = scores_all.shape[0] // 2

        labels = torch.eye(2*bs).to(device).reshape((-1,))
        scores = scores_all.reshape((-1,))

        loss = self.loss_fn(scores, labels).reshape((2*bs, 2*bs))*neg_mask

        loss_pos = loss.diag().mean()
        loss_neg = (loss.sum() - loss.diag().sum()) / (neg_mask.sum() - 2*bs)

        L = 0.5*loss_pos + 0.5*loss_neg

        return L

    def get_score_spatial(self, scores_raw, neg_mask, device=conf.training_param['device']):

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

        loss_semantic = self.get_score_semantic(scores_raw, neg_mask)
        loss_spatial = self.get_score_spatial(scores_raw, neg_mask)

        loss_total = 0.5*loss_spatial + 0.5*loss_semantic

        loss = {
            'total': loss_total,
            'easy': loss_semantic,
            'hard': loss_spatial,
        }

        loss_perfect = {'total': 0.00,
                        'random': -1*torch.log10(torch.tensor([0.5])) }

        return loss, loss_perfect # perfect loss here is 0.00 (because it is nn.BCE)


# class loss_AVOL_flip(nn.Module):

#     def __init__(self, eps=1e-8, contrast_param=conf.contrast_param):
#         super().__init__()
#         self.loss_fn = nn.BCELoss(reduction='none')
#         self.eps=eps
#         self.flip_neg = contrast_param['flip_img'] or contrast_param['flip_mic']

#     def get_score(self, scores_raw, neg_mask, alpha, device=conf.training_param['device']):

#         bs = scores_raw.shape[0]

#         bs_true = bs // 2

#         scores_easy = scores_raw[:bs_true, :bs_true]
#         scores_hard = scores_raw[bs_true:, bs_true:]
#         neg_mask = neg_mask[:bs_true, :bs_true]

#         l_easy = torch.eye(bs_true).to(device=device).view(-1,1)
#         l_hard = torch.concat((torch.ones((bs_true,1)), torch.zeros((bs_true, 1)))).to(device=device)

#         s_easy = scores_easy.reshape(-1,1)
#         s_hard = torch.concat((scores_easy.diag(), scores_hard.diag())).unsqueeze(-1)

#         scores = torch.concat((s_easy, s_hard))
#         labels = torch.concat((l_easy, l_hard))
        
#         loss = self.loss_fn(scores, labels)

#         ls_easy, ls_hard = loss.tensor_split((bs_true*bs_true, ), dim=0)

#         ls_easy = ls_easy.reshape((bs_true, bs_true))*neg_mask
#         ls_easy *= (torch.ones((bs_true,bs_true)) + (bs_true-1)*torch.eye(bs_true)).to(device=device)
#         ls_easy = ls_easy.sum() / (neg_mask.sum() + bs_true*(bs_true-1) + self.eps)

#         ls_hard = ls_hard.mean()

#         loss = {
#             'total': (1-alpha)*ls_easy + alpha*ls_hard,
#             'easy': ls_easy,
#             'hard': ls_hard,
#         }

#         return loss


#     def forward(self, scores_raw, neg_mask, alpha=conf.contrast_param['alpha'], device=conf.training_param['device']):

#         loss = self.get_score(scores_raw, neg_mask, alpha)

#         loss_perfect = {'total': 0.00 }

#         return loss, loss_perfect # perfect loss here is 0.00 (because it is nn.BCE)


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

        scores = rearrange(scores_raw, 'a1 a2 c -> (a1 a2) c')
        labels = rearrange(labels, 'a1 a2 c -> (a1 a2) c')

        if not return_err:

            loss = rearrange(self.loss_fn(scores, labels), '(a1 a2) -> a1 a2', a1=bs)
            loss = loss*neg_mask
            loss = (torch.ones((bs,bs)) + (bs-1)*torch.eye(bs)).to(device=device)*loss

            loss = loss.sum() / (neg_mask.sum() + self.eps)

            return loss

        else:

            _, idx1 = torch.max(scores, dim=-1)
            _, idx2 = torch.max(labels, dim=-1)

            idx1 = rearrange(idx1, '(a1 a2) -> a1 a2', a1=bs)
            idx2 = rearrange(idx2, '(a1 a2) -> a1 a2', a1=bs)

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

