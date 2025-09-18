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

        # KL Divergence Loss - Fixed to avoid NaN
        pred_scaled = y_pred_r / self.tau
        true_scaled = y_true_r / self.tau
        softmax_pred = F.softmax(pred_scaled, dim=-1)
        softmax_true = F.softmax(true_scaled, dim=-1)
        # Add small epsilon to avoid log(0)
        kl_loss = F.kl_div((softmax_pred + 1e-8).log(), softmax_true, reduction='batchmean')

        # Binary Cross Entropy Loss for classification
        bce_loss = self.criterion_c(y_pred_c, y_true_c)

        # L1 Loss Term - only for active periods
        l1_loss = torch.mean(torch.abs(y_pred_r - y_true_r) * y_true_c)

        # Combined Loss - balanced weights for better classification learning
        total_loss = mse_loss + 0.01 * kl_loss + 2.0 * bce_loss + self.lambda_ * l1_loss

        return total_loss
