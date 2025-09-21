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

        # KL Divergence Loss (with numerical stability)
        softmax_pred = F.softmax(y_pred_r / self.tau, dim=-1)
        softmax_true = F.softmax(y_true_r / self.tau, dim=-1)
        # Add small epsilon to prevent log(0)
        kl_loss = F.kl_div((softmax_pred + 1e-8).log(), softmax_true, reduction='batchmean')

        # Binary Cross Entropy Loss (more stable than soft-margin)
        bce_loss = self.criterion_c(y_pred_c, y_true_c)

        # L1 Loss Term (only for active appliances)
        l1_loss = torch.mean(torch.abs(y_pred_r - y_true_r) * y_true_c)

        # Combined Loss (removed unstable soft-margin loss)
        total_loss = mse_loss + kl_loss + bce_loss + self.lambda_ * l1_loss

        return total_loss
