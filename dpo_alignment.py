import pandas as pd
from transformers import AutoModel, AutoTokenizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM, GenerationConfig
from datasets import load_dataset
from trl import PPOTrainer, PPOConfig, AutoModelForSeq2SeqLMWithValueHead
from trl import create_reference_model
import torch
from peft import PeftModel, prepare_model_for_kbit_training, PeftConfig, get_peft_model, LoraConfig, TaskType
from transformers import BitsAndBytesConfig
from transformers import AutoModelForSeq2SeqLM
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, TaskType
from sklearn.model_selection import train_test_split
from datasets import Dataset
import os
from dataclasses import dataclass, field
from typing import Dict, Optional
import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, TrainingArguments
from trl import DPOTrainer
from transformers.models.bart.configuration_bart import BartConfig
from transformers import (
    BartTokenizerFast,
    AdamW
)
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments,TrainingArguments
from trl import DPOTrainer, AutoModelForSeq2SeqLMWithValueHead
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments,TrainingArguments
from trl import DPOTrainer, AutoModelForSeq2SeqLMWithValueHead



import os
import gc
import torch

import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from datasets import load_dataset
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from trl import DPOTrainer
import bitsandbytes as bnb
#from google.colab import userdata
import wandb



local_model_path="/mnt/Data/xx/x/x/x"
#local_model_path="facebook/bart-base"
#model = AutoModel.from_pretrained(loca)
#model_name="meta-llama/Llama-2-7b-hf"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    load_in_4bit=True,
    #device_map="auto"
)

#model.config.use_cache = False

# model = AutoModelForSeq2SeqLM.from_pretrained(
# local_model_path, # location of saved SFT model
# torch_dtype=torch.float32,
# low_cpu_mem_usage=True
# )

model_ref=model


peft_config = LoraConfig(
    r=16,
    lora_alpha=16,
    lora_dropout=0.06,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=['k_proj', 'gate_proj', 'v_proj', 'up_proj', 'q_proj', 'o_proj', 'down_proj']
)


train_data=pd.read_csv("train.csv")
valid_data=pd.read_csv("valid.csv")

def create_dict_from_dataframe(df):
    # Extract values from the specified columns
    prompt_values = df['xx'].tolist()
    chosen_values = df['xx'].tolist()
    rejected_values = df['xx'].tolist()

    # Create the dictionary
    result_dict = {'prompt': prompt_values, 'chosen': chosen_values, 'rejected': rejected_values}

    return result_dict


dpo_train_dataset_dict=create_dict_from_dataframe(train_data)
dpo_valid_dataset_dict=create_dict_from_dataframe(valid_data)

from datasets import Dataset
train_dataset = Dataset.from_dict(dpo_train_dataset_dict)
valid_dataset = Dataset.from_dict(dpo_valid_dataset_dict)

print(train_dataset)

tokenizer= AutoTokenizer.from_pretrained(local_model_path)
#ref_model=model
tokenizer.pad_token = tokenizer.eos_token

training_args = TrainingArguments(
                per_device_train_batch_size=8,
                per_device_eval_batch_size=8,
                remove_unused_columns=False,
                gradient_accumulation_steps=4,
                learning_rate=5e-7,
                evaluation_strategy="steps",
                max_steps=1000,
                logging_first_step=True,
                lr_scheduler_type="cosine",
                logging_steps=10,
                output_dir='/mnt/Data/xx/x/x/x',
                gradient_checkpointing=True,
#                 optim=torch.optim.Adam(model.parameters(), lr=2e-5)
               )

dpo_trainer = DPOTrainer(
    model,
    model_ref,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    tokenizer=tokenizer,
    beta=0.6,
    precompute_ref_log_probs=False,
    max_prompt_length=1000,
    max_length=1250,
    max_target_length=250,
   
  
)

#import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

dpo_trainer.train()

















