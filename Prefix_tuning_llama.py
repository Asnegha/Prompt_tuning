import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.nn import Module, Parameter
from torch.utils.data import DataLoader, Dataset

class PrefixTuning(Module):
    def __init__(self, config, num_prefix_tokens=20):
        super().__init__()
        self.num_prefix_tokens = num_prefix_tokens
        self.prefix_keys = Parameter(
            torch.randn(config.num_hidden_layers, num_prefix_tokens, config.hidden_size)
        )
        self.prefix_values = Parameter(
            torch.randn(config.num_hidden_layers, num_prefix_tokens, config.hidden_size)
        )

    def forward(self, past_key_values, batch_size):
        # Pre-expand prefix tensors once
        prefix_keys = self.prefix_keys.unsqueeze(1).expand(-1, batch_size, -1, -1)
        prefix_values = self.prefix_values.unsqueeze(1).expand(-1, batch_size, -1, -1)

        # Add prefix to each layer's past_key_values
        new_past_key_values = [
            (
                torch.cat([prefix_keys[layer], key], dim=1),
                torch.cat([prefix_values[layer], value], dim=1),
            )
            for layer, (key, value) in enumerate(past_key_values)
        ]
        return tuple(new_past_key_values)


# Load the model and tokenizer
model_name = "meta-llama/Meta-Llama-3.1-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name)
model.gradient_checkpointing_enable()


# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Freeze base model parameters
for param in model.parameters():
    param.requires_grad = False

# Add prefix tuning module
num_prefix_tokens = 20
prefix_tuning = PrefixTuning(model.config, num_prefix_tokens=num_prefix_tokens)

# Move prefix_tuning to the same device as model
prefix_tuning = prefix_tuning.to(device)

# Replace the past_key_values in the forward pass
def modified_forward(
    input_ids, attention_mask, past_key_values=None, **kwargs
):
    batch_size = input_ids.size(0)

    # If past_key_values is not provided, initialize it with empty tensors
    if past_key_values is None:
        num_layers = model.config.num_hidden_layers
        past_key_values = [
            (
                torch.zeros(batch_size, 0, model.config.hidden_size, device=input_ids.device),
                torch.zeros(batch_size, 0, model.config.hidden_size, device=input_ids.device),
            )
            for _ in range(num_layers)
        ]

    # Update past_key_values with prefix tuning
    past_key_values = prefix_tuning(past_key_values, batch_size)

    return model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        **kwargs
    )

# Bind the modified forward function to the model
model.forward = modified_forward

# Example dataset for instruction fine-tuning
class InstructionDataset(Dataset):
    def __init__(self, tokenizer, data, max_length=512):
        self.tokenizer = tokenizer
        self.data = data
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        instruction, response = self.data[idx]
        inputs = self.tokenizer(
            instruction,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        labels = self.tokenizer(
            response,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": labels["input_ids"].squeeze(0),
        }

# Sample data
sample_data = [
    ("Translate the following sentence to French: 'Hello, how are you?'",
     "Traduisez la phrase suivante en français : 'Bonjour, comment ça va ?'"),
    ("Summarize the following text: 'The quick brown fox jumps over the lazy dog.'",
     "The fox jumps over the dog."),
]

# Create dataset and dataloader
dataset = InstructionDataset(tokenizer, sample_data)
dataloader = DataLoader(dataset, batch_size=2)

# Example fine-tuning loop for instruction tuning
def fine_tune_instruction(model, tokenizer, train_dataloader, num_epochs=3):
    optimizer = torch.optim.Adam(prefix_tuning.parameters(), lr=5e-5)
    model.train()

    scaler = torch.cuda.amp.GradScaler()  # for automatic mixed precision

    for epoch in range(num_epochs):
        for batch in train_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():  # Enable mixed precision
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

            scaler.scale(loss).backward()  # Scale loss for FP16
            scaler.step(optimizer)
            scaler.update()

            print(f"Epoch {epoch + 1}, Loss: {loss.item()}")


fine_tune_instruction(model, tokenizer, dataloader, num_epochs=3)

# Save the fine-tuned prefix tuning parameters
prefix_tuning_save_path = "prefix_tuning_instruction_model.pt"
torch.save(prefix_tuning.state_dict(), prefix_tuning_save_path)
print(f"Prefix tuning embeddings saved to {prefix_tuning_save_path}")
