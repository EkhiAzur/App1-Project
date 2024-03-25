"""

Custom Trainer for contrastive learning

"""
from transformers import Trainer    
from pytorch_metric_learning.losses import SupConLoss


class ConstTrainer(Trainer):

    def __init__(self, temp, lam, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.temp = temp

        self.contrastive_loss = SupConLoss(temperature=self.temp)

        self.lam = lam

    def compute_loss(self, model, inputs, return_outputs=False):

        cross_loss, outputs = super().compute_loss(model, inputs, return_outputs=True)

        # Get CLS token embeddings
        embeddings = outputs["hidden_states"][-1][:, 0, :]
    
        labels = inputs["labels"]

        # Calculate SCL
        contrastive_loss = self.contrastive_loss(embeddings, labels)

        lam = self.lam

        # Combine losses
        loss = lam * cross_loss + (1- lam) * contrastive_loss

        return (loss, outputs) if return_outputs else loss
