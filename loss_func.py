import torch
import torch.nn as nn
import torch.nn.functional as F

class BERT4NILMLoss(nn.Module):
    def __init__(self, tau=0.1, lambda_=1.0):
        super(BERT4NILMLoss, self).__init__()
        self.tau = tau
        self.lambda_ = lambda_
        self.criterion_r = nn.MSELoss()
        self.criterion_c = nn.BCELoss()

    def forward(self, y_pred_r, y_true_r, y_pred_c, y_true_c):
        # Mean Squared Error Loss
        mse_loss = self.criterion_r(y_pred_r, y_true_r)
        
        # KL Divergence Loss
        softmax_pred = F.softmax(y_pred_r / self.tau, dim=-1)
        softmax_true = F.softmax(y_true_r / self.tau, dim=-1)
        kl_loss = F.kl_div(softmax_pred.log(), softmax_true, reduction='batchmean')
        
        # Soft-Margin Loss
        soft_margin_loss = torch.mean(torch.log(1 + torch.exp(-y_true_c * y_pred_c)))
        
        # L1 Loss Term
        l1_loss = torch.mean(torch.abs(y_pred_r - y_true_r) * (y_true_c > 0.5).float())

        # Combined Loss
        total_loss = mse_loss + kl_loss + soft_margin_loss + self.lambda_ * l1_loss
        
        return total_loss