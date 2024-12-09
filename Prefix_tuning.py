from transformers import AutoTokenizer, AutoModelForCausalLM, default_data_collator, get_linear_schedule_with_warmup,TrainingArguments
from peft import get_peft_config, get_peft_model, get_peft_model_state_dict, PrefixTuningConfig, TaskType
from trl import SFTConfig, SFTTrainer
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import random
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

device = "cuda"
model_name_or_path = "mistralai/Mistral-7B-v0.1"
tokenizer_name_or_path = "mistralai/Mistral-7B-v0.1"



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
sample_size = int(len(train_dataset) * 0.1)
indices = random.sample(range(len(train_dataset)), sample_size)

# Create the sampled dataset
train_dataset = train_dataset.select(indices)

# View sampled data
print(train_dataset)

train_dataset = train_dataset.map(lambda item: {'text': convert_to_alpaca(item)})
print(train_dataset[0]['text'])

model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
)

model.config.use_cache = False # silence the warnings
model.config.pretraining_tp = 1
# model.gradient_checkpointing_enable()

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_eos_token = True


peft_config = PrefixTuningConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, num_virtual_tokens=20)
model = get_peft_model(model, peft_config)
print(model.print_trainable_parameters())


training_arguments = TrainingArguments(
    save_strategy="steps",
    save_steps=500,
    output_dir="Hindi_pt_mistral",
    num_train_epochs=5,
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
    peft_config=peft_config,
    dataset_text_field="text",
    tokenizer=tokenizer,
    args=training_arguments,
    packing= False,
    max_seq_length=1024
    )
    

trainer.train()
    
peft_model_id = "snegha24/hindi_pt_mistral"
model.push_to_hub(peft_model_id, use_auth_token=True)