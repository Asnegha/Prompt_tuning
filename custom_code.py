import torch
import torch.nn as nn
from typing import Optional, Union, List, Tuple
from transformers.models.llama.modeling_llama import LlamaModel, LlamaDecoderLayer, LlamaPreTrainedModel, LlamaConfig, BaseModelOutputWithPast, GenerationMixin, CausalLMOutputWithPast
from transformers.cache_utils import Cache
from transformers.utils import logging
from transformers import AutoTokenizer, AutoModelForCausalLM, default_data_collator, get_linear_schedule_with_warmup,TrainingArguments, LlamaForCausalLM
from trl import SFTConfig, SFTTrainer
from peft import get_peft_config, get_peft_model, get_peft_model_state_dict, PromptTuningConfig, TaskType, PromptTuningInit
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import random
import os
from torch.nn import CrossEntropyLoss

device = 'cuda'

logger = logging.get_logger(__name__)

class PrefixTuning(nn.Module):
    def __init__(self, num_layers: int, prefix_length: int, hidden_size: int):
        super().__init__()
        self.num_layers = num_layers
        self.prefix_length = prefix_length
        self.hidden_size = hidden_size

        
        # Learnable prefix tokens for each layer
        self.prefix_tokens = nn.ParameterList([
            nn.Parameter(torch.randn(1, prefix_length, hidden_size))
            for _ in range(num_layers)
        ])
        
    def forward(self, layer_idx: int, batch_size: int):
        # Retrieve learnable prefix tokens for a specific layer
        prefix_output = self.prefix_tokens[layer_idx]
        prefix_output = prefix_output.expand(batch_size, -1, -1)  # (batch_size, prefix_length, hidden_size)
        return prefix_output

class PrefixTuningLlamaModel(LlamaModel):
    """
    LlamaModel with prefix tuning integrated into each layer. Prefix tokens are learned independently per layer.
    The base model is frozen, and only prefix parameters are trainable.
    """
    def __init__(self, config: LlamaConfig, prefix_length: int = 10):
        super().__init__(config)
        self.prefix_length = prefix_length
        self.hidden_size = config.hidden_size

        # Prefix tuning module
        self.prefix_tuning = PrefixTuning(
            num_layers=config.num_hidden_layers, 
            prefix_length=prefix_length, 
            hidden_size=config.hidden_size
        )

        # Freeze the base model parameters
        for param in self.parameters():
            param.requires_grad = False
        
        # Ensure prefix tuning parameters are trainable
        for param in self.prefix_tuning.parameters():
            param.requires_grad = True

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        batch_size = inputs_embeds.size(0)
        prefix_length = self.prefix_length
            
        # Generate prefix embeddings and position IDs
        prefix_embeds = self.prefix_tuning(0, batch_size)  # Layer 0 prefix tokens initially
        prefix_position_ids = torch.arange(0, prefix_length, device=inputs_embeds.device).unsqueeze(0)

        # Concatenate prefix tokens with input embeddings
        inputs_embeds1 = torch.cat((prefix_embeds, inputs_embeds), dim=1)
        
            
        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds1.shape[1], device=inputs_embeds.device
            )
        
            

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)
        
        # print(f'**********{inputs_embeds.shape, position_ids.shape}***********')
        

        position_embeddings = self.rotary_emb(inputs_embeds1, position_ids)
        
        # Attention mask update for prefix tokens
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, inputs_embeds1.size(1), device=inputs_embeds.device)
            # print(f'**********{inputs_embeds.shape, position_ids.shape,attention_mask.shape}***********')
        else:
            prefix_attention_mask = torch.ones(batch_size, prefix_length, device=attention_mask.device)
            attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1 ) 
            # print(f'**********{inputs_embeds.shape, position_ids.shape,attention_mask.shape}***********')
            
        
        # print(f'**********{inputs_embeds.shape, position_ids.shape,attention_mask.shape}***********')
        
        
        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds1, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds
        
        # Decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for layer_idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # Generate prefix tokens for this layer
            prefix_embeds = self.prefix_tuning(layer_idx, batch_size)
            hidden_states = torch.cat((prefix_embeds, hidden_states), dim=1)
            # print(f'**********{hidden_states.shape, position_ids.shape,attention_mask.shape}***********')
        

            # Pass through the decoder layer
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )

            hidden_states = layer_outputs[0]
            hidden_states = hidden_states[:, prefix_length:, :]


            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

class PrefixTuningLlamaCausalLM(LlamaForCausalLM):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}

    def __init__(self, config):
        super().__init__(config)
        self.model = PrefixTuningLlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
      


# Model instantiation
model_name_or_path = "meta-llama/Meta-Llama-3.1-8B"
model = PrefixTuningLlamaCausalLM.from_pretrained(
    model_name_or_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

def count_parameters(model):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    total_params = trainable_params + frozen_params

    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Frozen Parameters: {frozen_params:,}")
    print(f"Total Parameters: {total_params:,}")

count_parameters(model)

def convert_to_alpaca(item):
    if item['input']:
        return "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n" + f"### Instruction:\n{item['instruction']}\n\n### Input:\n{item['input']}\n\n### Response:\n{item['output']}"
    else:
        return "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n" + f"### Instruction:\n{item['instruction']}\n\n### Response:\n{item['output']}"


ds = load_dataset("MBZUAI/Bactrian-X", "hi")
train_dataset = ds['train']

# Fix seed for reproducibility
random.seed(42)

# Sample 10% of the data
sample_size = int(len(train_dataset) * 0.01)
indices = random.sample(range(len(train_dataset)), sample_size)

# Create the sampled dataset
train_dataset = train_dataset.select(indices)

# View sampled data
print(f'Train_dataset:{train_dataset}')

train_dataset = train_dataset.map(lambda item: {'text': convert_to_alpaca(item)})
print(f'Train_dataset:{train_dataset}')
print(f'Prompt: {type(train_dataset["text"])}')


# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_eos_token = True




training_arguments = TrainingArguments(
    save_strategy="steps",
    save_steps=500,
    output_dir="Hindi_pt",
    num_train_epochs=3,
    gradient_accumulation_steps = 1,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=1,
    lr_scheduler_type ='cosine',
    learning_rate=1e-6,
    weight_decay=0,
    warmup_steps = 10,
)

    # Setting sft parameters
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    dataset_text_field="text",
    tokenizer=tokenizer,
    args=training_arguments,
    packing= False,
    max_seq_length=2048
    )
    

trainer.train()
    
peft_model_id = "snegha24/hindi_pt"
model.push_to_hub(peft_model_id, use_auth_token=True)
