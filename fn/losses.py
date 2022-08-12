import torch
import torch.nn as nn
from einops import rearrange
import matplotlib.pyplot as plt


import core.config as conf


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


class loss_VSL_hard(nn.Module):

    def __init__(self, tau=0.2, eps=1e-4, alpha=0.5):
        super().__init__()
        self.tau=tau
        self.eps=eps
        self.alpha=alpha

    def get_score(self, scores_raw, neg_mask):

        scores_all = torch.exp(scores_raw/self.tau)*neg_mask

        sc_raw = scores_raw.cpu().detach().numpy()
        sc_all = scores_all.cpu().detach().numpy()

        # plt.figure()
        # plt.subplot(1,2,1)
        # plt.imshow(sc_raw)
        # plt.subplot(1,2,2)
        # plt.imshow(sc_all)
        # plt.show()

        bs = scores_all.shape[0] // 2

        scores_easy = scores_all[:bs, :bs]
        scores_hard = scores_all[bs:, bs:]

        s_pos = scores_easy.diag()
        s_hard = scores_hard.diag() + s_pos
        s_neg_a = scores_easy.sum(dim=1)
        s_neg_v = scores_easy.sum(dim=0) 
        
        L_av = (-1 * torch.log10((s_pos + self.eps) / (s_neg_a + self.eps))) . sum()
        L_va = (-1 * torch.log10((s_pos + self.eps) / (s_neg_v + self.eps))) . sum()
        L_hd = (-1 * torch.log10((s_pos + self.eps) / (bs*s_hard + self.eps))) . sum()

        norm_sum = self.alpha*neg_mask[:bs, :bs].sum() + (1-self.alpha)*bs*bs

        L = (self.alpha*(L_va+L_av) + (1-self.alpha)*L_hd) / (norm_sum + self.eps)

        return L


    def forward(self, scores_raw, neg_mask):

        bs = scores_raw.shape[0] // 2

        loss = self.get_score(scores_raw, neg_mask)

        zer = torch.zeros((bs,bs))
        one = torch.ones((bs, bs))
        ide = torch.eye(bs)

        labels = torch.cat((torch.cat((-1*one + 2*ide, zer), dim=0), torch.cat((zer, -1*ide), dim=0)), dim=1)
        
        loss_perfect = self.get_score(labels.to(device=conf.training_param['device']), neg_mask)

        return loss, loss_perfect


class loss_AVOL(nn.Module):

    def __init__(self, eps=1e-8):
        super().__init__()
        self.loss_fn = nn.BCELoss(reduction='none')
        self.eps=eps

    def get_score(self, scores_raw, neg_mask, device=conf.training_param['device']):

        bs = scores_raw.shape[0]

        labels = torch.eye(len(scores_raw)).to(device=device).view(-1,1)
        scores = scores_raw.view(-1,1)

        loss = self.loss_fn(scores, labels).view(bs,bs)
        loss = loss*neg_mask
        loss = (torch.ones((bs,bs)) + (bs-1)*torch.eye(bs)).to(device=device)*loss

        loss = loss.sum() / (neg_mask.sum() + self.eps)

        return loss


    def forward(self, scores_raw, neg_mask, device=conf.training_param['device'], return_err=False):

        loss = self.get_score(scores_raw, neg_mask)

        return loss, 0.00 # perfect loss here is 0.00 (because it is nn.BCE)


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

