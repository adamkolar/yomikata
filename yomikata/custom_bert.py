from typing import Optional, Union, Tuple
import torch
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import TokenClassifierOutput

from transformers import (
    BertForTokenClassification,
    DataCollatorForTokenClassification,
)

class CustomBertForTokenClassification(BertForTokenClassification):

    def __init__(self, config, alpha=None, gamma=2):
        super(CustomBertForTokenClassification, self).__init__(config)
        self.gamma = gamma
        self.alpha = alpha

    def to(self, *args, **kwargs):
        super(CustomBertForTokenClassification, self).to(*args, **kwargs)
        if self.alpha is not None:
            self.alpha = self.alpha.to(*args, **kwargs)
        return self

    def set_alpha(self, alpha):
        self.alpha = alpha.to(self.device)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        # modified section

        # masking out irrelevant classes

        if valid_mask != None:
            class_indices = torch.arange(logits.shape[-1], device=valid_mask.device).view(1, 1, -1)
            valid_mask_0 = (valid_mask >> 32).long()
            valid_mask_1 = (valid_mask & 0xFFFFFFFF).long()
            logits.add_(
                (
                    (class_indices < valid_mask_0.unsqueeze(-1)) | 
                    (class_indices >= valid_mask_1.unsqueeze(-1))
                    & (labels.unsqueeze(-1) != -100)
                ).float() * -1e9
            )

        # focal loss

        loss = None
        if labels is not None:
            inputs = logits.view(-1, self.num_labels)
            targets = labels.view(-1)
            loss_fct = CrossEntropyLoss(reduction='none')
            ce_loss = loss_fct(inputs, targets)
            pt = torch.exp(-ce_loss)
            mask = (targets != -100)
            loss_values = (self.alpha[targets[mask]] if self.alpha is not None else 1) * (1 - pt[mask]) ** self.gamma * ce_loss[mask]
            loss = loss_values.mean()
            loss *= 1e6

        # modified section : end

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class CustomDataCollatorForTokenClassification(DataCollatorForTokenClassification):
    def __call__(self, features):
        valid_masks = [feature.pop("valid_mask") for feature in features]
        batch = super().__call__(features)
        sequence_length = batch["input_ids"].shape[1]
        padding_side = self.tokenizer.padding_side
        if padding_side == "right":
            batch["valid_mask"] = [mask + [0] * (sequence_length - len(mask)) for mask in valid_masks]
        else:
            batch["valid_mask"] = [[0] * (sequence_length - len(mask) + mask) for mask in valid_masks]
        batch["valid_mask"] = torch.tensor(batch["valid_mask"], dtype=torch.int64)
        return batch
