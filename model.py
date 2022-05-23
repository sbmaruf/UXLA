import os
import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.nn import CrossEntropyLoss
from transformers import BertForTokenClassification, XLMRobertaForTokenClassification

class NegEntropy(object):
    def __call__(self,outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log()*probs, dim=1))
    

class Context_NER_BERT(BertForTokenClassification):
    r"""
        https://github.com/huggingface/transformers/blob/master/transformers/modeling_bert.py#L1122
    """
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None, 
                labels=None, augmented_logits=None, augmented_logits_lambda=.1, penalty=0, head_idx=0):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        
        if augmented_logits is not None:
            logits = augmented_logits_lambda * logits + (1-augmented_logits_lambda) * augmented_logits
        
        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            conf_penalty = NegEntropy()
            loss_fct_token = CrossEntropyLoss(reduction='none')
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
                if penalty == 1 :
                    loss = loss - conf_penalty(active_logits)
                elif penalty == 2:
                    loss = loss + conf_penalty(active_logits)
                per_token_loss = loss_fct_token(logits.view(-1, self.num_labels), labels.view(-1))
                
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                if penalty == 1 :
                    loss = loss - conf_penalty(logits.view(-1, self.num_labels))
                elif penalty == 2:
                    loss = loss + conf_penalty(logits.view(-1, self.num_labels))
                per_token_loss = loss_fct_token(logits.view(-1, self.num_labels), labels.view(-1))
                
            outputs = (loss,) + outputs

        return outputs,  per_token_loss # (loss), scores, (hidden_states), (attentions)


class Context_NER_XLMR(XLMRobertaForTokenClassification):
    r"""
        https://github.com/huggingface/transformers/blob/master/transformers/modeling_bert.py#L1122
    """         
    def forward(
            self, 
            input_ids=None, 
            attention_mask=None, 
            token_type_ids=None,
            position_ids=None, 
            head_mask=None, 
            inputs_embeds=None, 
            labels=None, 
            penalty=0,
            head_idx=0
        ):

        outputs = self.roberta(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        
        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            conf_penalty = NegEntropy()
            loss_fct_token = CrossEntropyLoss(reduction='none')
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
                if penalty == 1 :
                    loss = loss - conf_penalty(active_logits)
                elif penalty == 2:
                    loss = loss + conf_penalty(active_logits)
                per_token_loss = loss_fct_token(logits.view(-1, self.num_labels), labels.view(-1))
                
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                if penalty == 1 :
                    loss = loss - conf_penalty(logits.view(-1, self.num_labels))
                elif penalty == 2:
                    loss = loss + conf_penalty(logits.view(-1, self.num_labels))
                per_token_loss = loss_fct_token(logits.view(-1, self.num_labels), labels.view(-1))
                
            outputs = (loss,) + outputs

        return outputs,  per_token_loss # (loss), scores, (hidden_states), (attentions)
 
    
from transformers.configuration_xlm_roberta import XLMRobertaConfig
from transformers import BertPreTrainedModel
XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP = {
    "xlm-roberta-base": "https://cdn.huggingface.co/xlm-roberta-base-pytorch_model.bin",
    "xlm-roberta-large": "https://cdn.huggingface.co/xlm-roberta-large-pytorch_model.bin",
    "xlm-roberta-large-finetuned-conll02-dutch": "https://cdn.huggingface.co/xlm-roberta-large-finetuned-conll02-dutch-pytorch_model.bin",
    "xlm-roberta-large-finetuned-conll02-spanish": "https://cdn.huggingface.co/xlm-roberta-large-finetuned-conll02-spanish-pytorch_model.bin",
    "xlm-roberta-large-finetuned-conll03-english": "https://cdn.huggingface.co/xlm-roberta-large-finetuned-conll03-english-pytorch_model.bin",
    "xlm-roberta-large-finetuned-conll03-german": "https://cdn.huggingface.co/xlm-roberta-large-finetuned-conll03-german-pytorch_model.bin",
}

from transformers.modeling_roberta import RobertaModel

class MlpTagger(nn.Module):
    def __init__(self,
                 num_layers,
                 input_size,
                 hidden_size,
                 output_size,
                 dropout,
                 batch_norm=False):
        super(MlpTagger, self).__init__()
        assert num_layers >= 0, 'Invalid layer numbers'
        self.hidden_size = hidden_size
        self.net = nn.Sequential()
        for i in range(num_layers):
            if dropout > 0:
                self.net.add_module('p-dropout-{}'.format(i), nn.Dropout(p=dropout))
            if i == 0:
                self.net.add_module('p-linear-{}'.format(i), nn.Linear(input_size, hidden_size))
            else:
                self.net.add_module('p-linear-{}'.format(i), nn.Linear(hidden_size, hidden_size))
            if batch_norm:
                self.net.add_module('p-bn-{}'.format(i), nn.LayerNorm(hidden_size))
            self.net.add_module('p-relu-{}'.format(i), nn.ReLU())

        self.net.add_module('p-linear-final', nn.Linear(hidden_size, output_size))
        # self.net.add_module('p-logsoftmax', nn.LogSoftmax(dim=-1))

    def forward(self, input):
        return self.net(input)


class MixtureOfExperts(nn.Module):
    def __init__(self,
                 num_layers,
                 input_size,
                 num_experts,
                 hidden_size,
                 output_size,
                 dropout,
                 bn=False):
        super(MixtureOfExperts, self).__init__()
        self.num_experts = num_experts
        self.gates = nn.Linear(input_size, num_experts)
        mlp = MlpTagger
        self.experts = nn.ModuleList([mlp(num_layers, \
                input_size, hidden_size, output_size, dropout, bn) \
                for _ in range(num_experts)])

    def forward(self, input):
        # input: bs x seqlen x input_size
        gate_input = input # input.detach() if opt.detach_gate_input else input
        gate_outs = self.gates(gate_input)
        gate_softmax = functional.softmax(gate_outs, dim=-1) # bs x seqlen x #experts
        # bs x seqlen x #experts x output_size
        expert_outs = torch.stack([exp(input) for exp in self.experts], dim=-2)
        # bs x seqlen x output_size
        output = torch.sum(gate_softmax.unsqueeze(-1) * expert_outs, dim=-2)
        # output logits
        # return output, gate_outs
        return output
    

class Context_NER_XLMR_Multi_Head(BertPreTrainedModel):
    r"""
        https://github.com/huggingface/transformers/blob/master/transformers/modeling_bert.py#L1122
    """
    config_class = XLMRobertaConfig
    pretrained_model_archive_map = XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.roberta = RobertaModel(config)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        # self.classifier = torch.nn.Linear(config.hidden_size, config.num_labels)
        self.classifier = MixtureOfExperts(
                                    num_layers=2,
                                    input_size=config.hidden_size,
                                    num_experts=3,
                                    hidden_size=config.hidden_size//4,
                                    output_size=config.num_labels,
                                    dropout=config.hidden_dropout_prob
                                )
        for i in range(config.num_of_heads-1):
            # setattr(self, "classifier_{}".format(i+1), torch.nn.Linear(config.hidden_size, config.num_labels))
            setattr(
                self, 
                "classifier_{}".format(i+1), 
                MixtureOfExperts(
                        num_layers=2,
                        input_size=config.hidden_size,
                        num_experts=3,
                        hidden_size=config.hidden_size//4,
                        output_size=config.num_labels,
                        dropout=config.hidden_dropout_prob
                    )
            )
        self.init_weights()

    def forward(
            self, 
            input_ids=None, 
            attention_mask=None, 
            token_type_ids=None,
            position_ids=None, 
            head_mask=None, 
            inputs_embeds=None, 
            labels=None, 
            penalty=0,
            head_idx=0
        ):

        outputs = self.roberta(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        if head_idx == 0 :
            logits = self.classifier(sequence_output)
        else:
            logits = getattr(self, "classifier_{}".format(head_idx))(sequence_output)
        
        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            conf_penalty = NegEntropy()
            loss_fct_token = CrossEntropyLoss(reduction='none')
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
                if penalty == 1 :
                    loss = loss - conf_penalty(active_logits)
                elif penalty == 2:
                    loss = loss + conf_penalty(active_logits)
                per_token_loss = loss_fct_token(logits.view(-1, self.num_labels), labels.view(-1))
                
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                if penalty == 1 :
                    loss = loss - conf_penalty(logits.view(-1, self.num_labels))
                elif penalty == 2:
                    loss = loss + conf_penalty(logits.view(-1, self.num_labels))
                per_token_loss = loss_fct_token(logits.view(-1, self.num_labels), labels.view(-1))
                
            outputs = (loss,) + outputs

        return outputs,  per_token_loss # (loss), scores, (hidden_states), (attentions)


def load_model(
        model_type, MODEL_CLASSES, 
        model_name_or_path, config_name, tokenizer_name, 
        num_labels, cache_dir, do_lower_case, device, dropout=.1,
        num_of_heads = 1
    ):
    model_type = model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[model_type]
    if num_of_heads > 1:
        model_class = Context_NER_XLMR_Multi_Head
    config = config_class.from_pretrained(config_name if config_name else model_name_or_path,
                                          num_labels=num_labels,
                                          cache_dir=cache_dir if cache_dir else None)
    config.hidden_dropout_prob = dropout
    config.num_of_heads = num_of_heads
    tokenizer = tokenizer_class.from_pretrained(tokenizer_name if tokenizer_name else model_name_or_path,
                                                do_lower_case=do_lower_case,
                                                cache_dir=cache_dir if cache_dir else None)
    model = model_class.from_pretrained(model_name_or_path,
                                        from_tf=bool(".ckpt" in model_name_or_path),
                                        config=config,
                                        cache_dir=cache_dir if cache_dir else None)
    model.to(device)

    return config, tokenizer, model


def save_model_checkpoint(
        args, output_dir, name, model, 
        logger=None, 
        checkpoint="best_dev_model", 
        overwrite_address = None,
        num_of_heads = 1
    ):
    if overwrite_address is None:
        name = name.replace(";", "_")
        output_dir = os.path.join(output_dir, "{}.{}".format(checkpoint, name))
    else:
        output_dir = overwrite_address
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model_to_save = model.module if hasattr(model, "module") else model 
    model_to_save.save_pretrained(output_dir)
    torch.save(args, os.path.join(output_dir, "training_args.bin"))

    if logger is None:
        print("Saving model checkpoint to {}".format(output_dir))
    else:
        logger.info("Saving model checkpoint to {}".format(output_dir))
    return output_dir