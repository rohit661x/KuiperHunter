import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        """
        Focal Loss for binary classification.
        
        Args:
            alpha (float): Weighting factor for the rare class (foreground).
            gamma (float): Focusing parameter for hard examples.
            reduction (str): 'mean', 'sum' or 'none'.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: Logits (B, C, ...).
            targets: Binary targets (B, C, ...).
        """
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        
        # Manual calculation of modulation
        # If target=1, we want alpha * (1-pt)^gamma * log(pt)
        # If target=0, we want (1-alpha) * pt^gamma * log(1-pt)
        # This is equivalent to: alpha_t * (1-pt_t)^gamma * CE
        
        loss = self.alpha * (1 - pt)**self.gamma * bce_loss * targets + \
               (1 - self.alpha) * pt**self.gamma * bce_loss * (1 - targets)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
