from unsloth import FastLanguageModel
import os
import torch
import pandas as pd
from datasets import Dataset
from trl import SFTTrainer, SFTConfig

os.environ["TOKENIZERS_PARALLELISM"] = "false"

max_seq_length = 4048
load_in_4bit = True


alpaca_instruction = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that strictly follows the requirements.

### Instruction:
Du bist ein klinischer Kodierungsassistent.

Gegeben eine Krankenhausentlassungszusammenfassung:
- Extrahiere die relevanten ICD-10-CM Codes.

WICHTIG:
- Antworte ausschliesslich mit einer JSON-Liste von Strings.
- Keine Erklärungen.
- Kein zusätzlicher Text.
- Keine Kommentare.
- Kein Markdown.
- Kein Codeblock.
- Nur die reine JSON-Liste.

Beispiel für das korrekte Ausgabeformat:
["F03.0","F03.1","G30.89","G40.89"]

### input:
{}

### Response:
{}"""


model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen3-14B-unsloth-bnb-4bit",
    load_in_4bit = load_in_4bit, )

peft_model = FastLanguageModel.get_peft_model(
    model,
    r=8,

    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj", ],
    lora_alpha=8,
    lora_dropout=0.0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=1234,
    use_rslora=False,
    loftq_config=None,
)


train_id_path = 'ids_training.txt'
dev_id_path = 'ids_development.txt'
code_path = 'anns_train_dev.txt'

with open(train_id_path) as f:
    train_ids = [x.rstrip() for x in f.readlines()]

with open(dev_id_path) as f:
    dev_ids = [x.rstrip() for x in f.readlines()]

with open(code_path) as f:
    code_ids = [x.rstrip() for x in f.readlines()]
    id_to_code = {kv.split('\t')[0]:kv.split('\t')[1].split('|') for kv in code_ids}

files = os.listdir('docs-training')


def formatting_dataset(id_path):
    EOS_TOKEN = tokenizer.eos_token
    texts = []

    with open(id_path) as f:
        ids = [x.rstrip() for x in f.readlines()]
    for t_id in ids:
        if '%s.txt' % t_id in files:
            with open(os.path.join('docs-training', '%s.txt' % t_id), 'r') as f:
                text=f.read()
                if t_id in id_to_code:
                    icd_code = id_to_code[t_id]
                    full_text = alpaca_instruction.format(text, icd_code) + EOS_TOKEN
                    texts.append(full_text)
                else:
                    pass
    return Dataset.from_dict({"text": texts, })

train_dataset = formatting_dataset(train_id_path)



sftConfig = SFTConfig(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_steps=5,
    num_train_epochs=4,
    learning_rate=2e-4,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    logging_steps=1,
    optim="adamw_8bit",
    weight_decay=0.03,
    lr_scheduler_type="linear",
    seed=1234,
    output_dir="outputs_icd",
)

trainer = SFTTrainer(
    model=peft_model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,
    args=sftConfig
)

trainer.train()


peft_model.save_pretrained_gguf(
    "/data/agha/gguf_ollama/icd_model_gguf",
    tokenizer,
    quantization_method="q4_k_m"
)




# When done, run this: ollama create qwen3_14b_icd_decoder -f /data/agha/gguf_ollama/icd_model_gguf_gguf/Modelfile
