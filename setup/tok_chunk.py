import os
from dotenv import load_dotenv
from transformers import AutoTokenizer

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
tokenizer = AutoTokenizer.from_pretrained("Talking-Babies/opt-tokenizer")


def tokenize_and_chunk(examples, seq_len):
    """
    Tokenizes and chunks text data to fixed-length sequences.
    
    Args:
        examples: A batch of text examples from the dataset
        seq_len: The length of the sequences to chunk the text into
        
    Returns:
        Dictionary containing chunked token sequences of length SEQ_LEN
    """
    tokens = []
    # Process each text example in the batch
    for text in examples['text']:
        # Convert text to token IDs
        _tokens = tokenizer.encode(text)
        # Add EOS token to mark the end of each text example
        _tokens.append(tokenizer.eos_token_id)
        # Accumulate all tokens in a flat list
        tokens.extend(_tokens)

    # Split the accumulated tokens into chunks of SEQ_LEN
    chunks = [tokens[i:i + seq_len] for i in range(0, len(tokens), seq_len)]
    
    # Discard the last chunk if it's shorter than SEQ_LEN to ensure uniform sequence length
    if len(chunks[-1]) < seq_len:
        chunks = chunks[:-1]
        
    return {'input_ids': chunks}

from datasets import Dataset, load_dataset
from huggingface_hub import HfApi
from functools import partial

# Load dataset from Hugging Face
dataset = load_dataset("tafseer-nayeem/KidLM-corpus", split="train")

# Assuming the dataset contains lines of text, and documents are separated by empty lines
raw_data_list = []

document = []
for example in dataset:
    line = example['text'].strip()
    if line == '':
        if document:
            raw_data_list.append({'text': ' '.join(document)})
            document = []
    else:
        document.append(line)

# Add the last document if needed
if document:
    raw_data_list.append({'text': ' '.join(document)})

# Convert to Hugging Face Dataset and shuffle
raw_dataset = Dataset.from_list(raw_data_list)
raw_dataset = raw_dataset.shuffle(seed=420)

# Optional: huggingface hub setup
api = HfApi()

# Flag
SINGLE_SHUFFLE = True 

from datasets import Dataset, load_dataset
from huggingface_hub import HfApi
from functools import partial
import os

# Load and clean KidLM corpus from Hugging Face
dataset = load_dataset("tafseer-nayeem/KidLM-corpus", split="train")

# Group lines into documents separated by empty lines
raw_data_list = []
document = []
for example in dataset:
    line = example['text'].strip()
    if line == '':
        if document:
            raw_data_list.append({'text': ' '.join(document)})
            document = []
    else:
        document.append(line)
if document:
    raw_data_list.append({'text': ' '.join(document)})

raw_dataset = Dataset.from_list(raw_data_list)
raw_dataset = raw_dataset.shuffle(seed=420)

# === Tokenization ===
# Make sure this function is defined elsewhere in your script
# def tokenize_and_chunk(example, seq_len): ...

tokenize_and_chunk_2048 = partial(tokenize_and_chunk, seq_len=2048)

tokenized_dataset_2048 = raw_dataset.map(
    tokenize_and_chunk_2048,
    batched=True,
    batch_size=500,
    num_proc=8,
    remove_columns=raw_dataset.column_names
)

# Optional final shuffle – this is if we want to have single or doubly shuffled data
SINGLE_SHUFFLE = False  # toggle as needed
if not SINGLE_SHUFFLE:
    tokenized_dataset_2048 = tokenized_dataset_2048.shuffle(seed=42)

# === Save to Parquet ===
os.makedirs("data/processed", exist_ok=True)
parquet_path = 'data/processed/train_100M_2048'
if SINGLE_SHUFFLE:
    parquet_path += '_single_shuffle'
parquet_path += '.parquet'

tokenized_dataset_2048.to_parquet(parquet_path)

# === Upload to Hugging Face Hub ===
HF_TOKEN = os.getenv("HF_TOKEN")  # ensure this is set in your environment
repo_id = "Talking-Babies/train_100M_2048"
if SINGLE_SHUFFLE:
    repo_id += "_single_shuffle"

api = HfApi()
api.create_repo(repo_id, private=False, exist_ok=True, token=HF_TOKEN, repo_type="dataset")

api.upload_file(
    path_or_fileobj=parquet_path,
    repo_id=repo_id,
    repo_type='dataset',
    path_in_repo='train_100M_2048_single_shuffle.parquet' if SINGLE_SHUFFLE else 'train_100M_2048.parquet',
    token=HF_TOKEN
)
