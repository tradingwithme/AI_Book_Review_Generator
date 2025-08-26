from google.colab import drive
drive.mount('/content/drive')
from json import load
with open('/content/drive/MyDrive/bookSummaries.json','r',encoding='utf-16') as file: bookSummaries = load(file)
with open('/content/drive/MyDrive/bookReviews.json','r',encoding='utf-16') as file: reviewTexts2 = load(file)
import pandas as pd
bookContent = pd.DataFrame(bookSummaries,columns=['bookTitle','bookContent'])
bookContent['bookTitle'] = bookContent['bookTitle'].apply(sanitizeText)
bookContent = bookContent[bookContent['bookContent'].apply(lambda x: len(str(x).split())>10)].reset_index(drop=True)
bookContent.drop_duplicates().reset_index(drop=True,inplace=True)
bookContent['bookContent'] = bookContent['bookContent'].apply(lambda x: x.replace('Back to store\n',''))
bookReviews = pd.DataFrame(reviewTexts2,columns=['bookTitle','reviews'])
bookReviews['bookTitle'] = bookReviews['bookTitle'].apply(sanitizeText)
merged_books_df = pd.merge(bookReviews, bookContent, on='bookTitle', how='inner')
display(merged_books_df.head())
import spacy
from transformers import AutoTokenizer
import pandas as pd

# Load spaCy English model for sentence segmentation
nlp = spacy.load("en_core_web_sm")

# Load the tokenizer for the model
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")

# Define a function to segment sentences using spaCy
def segment_sentences(book_content):
    doc = nlp(book_content)  # Process the content with spaCy
    sentences = [sent.text.strip() for sent in doc.sents]  # Extract sentences
    return sentences

# Function to chunk content into smaller pieces based on token limit
def chunk_sentences(sentences, tokenizer, max_length=2048):
    chunks = []
    current_chunk = []
    current_length = 0

    # Iterate over the segmented sentences and accumulate tokens
    for sentence in sentences:
        tokenized_sentence = tokenizer.encode(sentence)
        token_length = len(tokenized_sentence)

        # If adding this sentence exceeds the max length, start a new chunk
        if current_length + token_length <= max_length:
            current_chunk.append(sentence)
            current_length += token_length
        else:
            # Store the current chunk and reset
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_length = token_length

    # Add the last chunk if necessary
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

# Apply sentence segmentation and chunking to the dataframe
merged_books_df['segmented_sentences'] = merged_books_df['bookContent'].apply(segment_sentences)

# Apply chunking to create chunks within the token limit
merged_books_df['chunks'] = merged_books_df['segmented_sentences'].apply(lambda x: chunk_sentences(x, tokenizer, max_length=2048))

# Display the chunks (example output)
print(merged_books_df[['bookTitle', 'chunks']].head())
from torch.optim import AdamW
from transformers import TrainingArguments, Adafactor
from transformers import Adafactor, TrainingArguments, Trainer

# Set the pad_token to eos_token
tokenizer.pad_token = tokenizer.eos_token

# Tokenize the chunked dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

chunked_dataset = chunked_dataset.map(tokenize_function, batched=True)

# Add the 'labels' column which is equal to 'input_ids' for language modeling
chunked_dataset = chunked_dataset.map(lambda x: {"labels": x["input_ids"]})

# Ensure the tokenizer outputs are in the right format
chunked_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# Define the optimizer
optimizer = Adafactor(
    model.parameters(),
    lr=1e-5,                    # Learning rate
    eps=(1e-30, 1e-3),          # Set eps as a tuple (recommended by Adafactor)
    weight_decay=0.01,          # Weight decay
    relative_step=False,         # Enable relative step (you can change it based on your needs)
    #warmup_init=True,           # Use warm-up initialization
)

# Define the training arguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    logging_dir="./logs",
    logging_steps=10,
    report_to="wandb",
    gradient_checkpointing=True,
    fp16=True,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=chunked_dataset,
    eval_dataset=chunked_dataset,
    tokenizer=tokenizer,
    optimizers=(optimizer, None),
)

# Start training
trainer.train()
# Create a function to generate the input prompt and target output (the review)
def create_prompt(title, review):
    prompt = f"Write a detailed review of the book titled '{title}'. The review should cover plot, themes, characters, and overall impression."
    return prompt, review  # The target is the review itself

# Apply this to the dataframe to create prompts and corresponding labels
merged_books_df['input'] = merged_books_df.apply(lambda row: create_prompt(row['bookTitle'], row['reviews'])[0], axis=1)
merged_books_df['labels'] = merged_books_df.apply(lambda row: create_prompt(row['bookTitle'], row['reviews'])[1], axis=1)

from transformers import GPT2Tokenizer

# Load the tokenizer for GPT-2
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Tokenization function
def tokenize_function(examples):
    return tokenizer(examples['input'], truncation=True, padding="max_length", max_length=512)

# Convert the DataFrame to a Hugging Face Dataset
from datasets import Dataset

# Convert to Hugging Face Dataset
chunked_dataset = Dataset.from_pandas(merged_books_df[['input', 'labels']])

# Tokenize the dataset
chunked_dataset = chunked_dataset.map(tokenize_function, batched=True)

# Ensure the tokenizer outputs are in the right format
chunked_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",          # Output directory
    num_train_epochs=3,              # Number of training epochs
    per_device_train_batch_size=4,   # Batch size
    per_device_eval_batch_size=4,    # Batch size for evaluation
    warmup_steps=500,                # Warmup steps
    weight_decay=0.01,               # Weight decay
    logging_dir="./logs",            # Logging directory
    logging_steps=10,
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=chunked_dataset,  # The tokenized training dataset
    eval_dataset=chunked_dataset,   # The evaluation dataset
    tokenizer=tokenizer,            # Tokenizer
)

# Start training
trainer.train()

# Example prompt to generate a review
prompt = "Write a detailed review of the book titled 'Toxic Turkey'. The review should cover plot, themes, characters, and overall impression."

# Tokenize the prompt
input_ids = tokenizer.encode(prompt, return_tensors="pt")

# Generate the review (temperature, max_length etc. can be adjusted)
generated_ids = model.generate(input_ids, max_length=512, temperature=0.7, num_return_sequences=1)

# Decode and print the generated review
generated_review = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
print(generated_review)

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load Sentence-BERT model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Assuming `merged_books_df['bookContent']` contains book content
book_contents = merged_books_df['bookContent'].tolist()

# Generate embeddings for the book content
embeddings = model.encode(book_contents, show_progress_bar=True)

# Convert embeddings to numpy array for Faiss index
embeddings = np.array(embeddings).astype('float32')

# Create a Faiss index
index = faiss.IndexFlatL2(embeddings.shape[1])  # L2 distance index
index.add(embeddings)  # Add embeddings to the index

# Create a new column in the dataframe for structured prompts + retrieved content
def create_input(title, retrieved_content):
    prompt = f"Write a detailed review of the book titled '{title}'. The review should cover plot, themes, characters, and overall impression."
    return prompt + "\n\n" + retrieved_content[0]  # Take the first retrieved content for simplicity

# Apply to all rows
merged_books_df['input'] = merged_books_df.apply(lambda row: create_input(row['bookTitle'], retrieve_content(row['bookTitle'])), axis=1)
from transformers import Trainer, TrainingArguments, GPT2LMHeadModel, GPT2Tokenizer

# Initialize model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Tokenize inputs and outputs
def tokenize_function(examples):
    return tokenizer(examples['input'], truncation=True, padding="max_length", max_length=512)

# Assuming 'merged_books_df' has the 'reviews' column with the target outputs (reviews)
chunked_dataset = Dataset.from_pandas(merged_books_df[['input', 'reviews']])

# Apply tokenization
chunked_dataset = chunked_dataset.map(tokenize_function, batched=True)

# Set dataset format
chunked_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# Define the training arguments
training_args = TrainingArguments(
    output_dir="./results",          # Output directory
    num_train_epochs=3,              # Number of training epochs
    per_device_train_batch_size=4,   # Batch size
    per_device_eval_batch_size=4,    # Batch size for evaluation
    warmup_steps=500,                # Warmup steps
    weight_decay=0.01,               # Weight decay
    logging_dir="./logs",            # Logging directory
    logging_steps=10,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=chunked_dataset,  # The dataset with tokenized inputs
    eval_dataset=chunked_dataset,   # The evaluation dataset
    tokenizer=tokenizer,            # Tokenizer
)

# Start training
trainer.train()
# Create the structured text-to-text pairs
  def create_text_to_text_pair(title, review):
      prompt = f"Write a detailed review of the book titled '{title}'. The review should cover plot, themes, characters, and overall impression."
      return prompt, review

  # Apply to dataframe
  merged_books_df['input'] = merged_books_df.apply(lambda row: create_text_to_text_pair(row['bookTitle'], row['reviews'])[0], axis=1)
  merged_books_df['labels'] = merged_books_df.apply(lambda row: create_text_to_text_pair(row['bookTitle'], row['reviews'])[1], axis=1)

from transformers import T5Tokenizer

  # Use T5 tokenizer for text-to-text tasks (you can also use other models like BART, etc.)
  tokenizer = T5Tokenizer.from_pretrained("t5-small")

  # Tokenization function for input-output text pairs
  def tokenize_function(examples):
      inputs = tokenizer(examples['input'], padding="max_length", truncation=True, max_length=512)
      labels = tokenizer(examples['labels'], padding="max_length", truncation=True, max_length=512)
      inputs['labels'] = labels['input_ids']  # Make sure labels are properly formatted
      return inputs

  # Convert the DataFrame to a Hugging Face Dataset
  from datasets import Dataset

  chunked_dataset = Dataset.from_pandas(merged_books_df[['input', 'labels']])

  # Tokenize the dataset
  chunked_dataset = chunked_dataset.map(tokenize_function, batched=True)

  # Ensure correct format for training
  chunked_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

  # Define training arguments
  training_args = TrainingArguments(
      output_dir="./results",          # Output directory
      num_train_epochs=3,              # Number of training epochs
      per_device_train_batch_size=4,   # Batch size
      per_device_eval_batch_size=4,    # Batch size for evaluation
      warmup_steps=500,                # Warmup steps
      weight_decay=0.01,               # Weight decay
      logging_dir="./logs",            # Logging directory
      logging_steps=10,
  )

  # Initialize the Trainer
  trainer = Trainer(
      model=model,
      args=training_args,
      train_dataset=chunked_dataset,  # The tokenized training dataset
      eval_dataset=chunked_dataset,   # The evaluation dataset
      tokenizer=tokenizer,            # Tokenizer
  )

  # Start training
  trainer.train()

#==========================================================================================================================
from google.colab import drive
drive.mount('/content/drive')
from json import load
import pandas as pd
import spacy
from transformers import T5Tokenizer, TrainingArguments, T5ForConditionalGeneration, Trainer
from datasets import Dataset
import torch
import time # Import the time module for prompting

# Define sanitizeText function
def sanitizeText(text):
    return str(text).strip().lower()

# Load data and create dataframes
with open('/content/drive/MyDrive/bookSummaries.json','r',encoding='utf-16') as file: bookSummaries = load(file)
with open('/content/drive/MyDrive/bookReviews.json','r',encoding='utf-16') as file: reviewTexts2 = load(file)

bookContent = pd.DataFrame(bookSummaries,columns=['bookTitle','bookContent'])
bookContent['bookTitle'] = bookContent['bookTitle'].apply(sanitizeText)
bookContent = bookContent[bookContent['bookContent'].apply(lambda x: len(str(x).split())>10)].reset_index(drop=True)
bookContent.drop_duplicates().reset_index(drop=True,inplace=True)
bookContent['bookContent'] = bookContent['bookContent'].apply(lambda x: x.replace('Back to store\n',''))

bookReviews = pd.DataFrame(reviewTexts2,columns=['bookTitle','reviews'])
bookReviews['bookTitle'] = bookReviews['bookTitle'].apply(sanitizeText)

merged_books_df = pd.merge(bookReviews, bookContent, on='bookTitle', how='inner')

# Detect device
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
else:
    try:
        import torch_xla.core.xla_model as xm
        device = xm.xla_device()
    except ImportError:
        pass # TPU not available

print(f"Detected device: {device}")

# Define device-specific parameters
if device == 'cpu':
    num_samples = 10  # Updated sample size for CPU
    num_epochs = 3
elif device == 'cuda':
    num_samples = 30 # Three times the CPU sample size
    num_epochs = 3 # Keeping epochs consistent with previous successful run
else:  # TPU device (assuming 'xla')
    num_samples = len(merged_books_df) # Use the full dataset for TPU
    num_epochs = 3 # Keeping epochs consistent with previous successful run

print(f"Number of samples for training: {num_samples}")
print(f"Number of epochs for training: {num_epochs}")

# Load the pre-trained T5 model and tokenizer (single instance for all training styles)
# Using 't5-small' as it was used in previous successful attempts
model = T5ForConditionalGeneration.from_pretrained("t5-small")
tokenizer = T5Tokenizer.from_pretrained("t5-small")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Create structured text-to-text pairs for input and labels (data preparation done once)
def create_text_to_text_pair(title, review):
    prompt = f"Write a detailed review of the book titled '{title}'. The review should cover plot, themes, characters, and overall impression."
    return prompt, review

# Apply to dataframe
merged_books_df['input'] = merged_books_df.apply(lambda row: create_text_to_text_pair(row['bookTitle'], row['reviews'])[0], axis=1)
merged_books_df['labels'] = merged_books_df.apply(lambda row: create_text_to_text_pair(row['bookTitle'], row['reviews'])[1], axis=1)


# Prompt the user for overall continuation
user_input = input(f"Detected device: {device}. Do you want to proceed with the sequential training for all three styles? (yes/no): ").lower()

if user_input in ['yes', 'y']:
    print("Proceeding with sequential training.")

    # --- Regular Training (using T5 for a text generation task) ---
    print("\n--- Starting Regular Training ---")

    # Convert the DataFrame to a Hugging Face Dataset for regular training (re-sample/slice)
    chunked_dataset_regular = Dataset.from_pandas(merged_books_df[['input', 'labels']])
    chunked_dataset_regular = chunked_dataset_regular.select(range(num_samples))

    # Tokenization function for regular training (using T5 tokenizer)
    def tokenize_function_regular(examples):
        inputs = tokenizer(examples['input'], padding="max_length", truncation=True, max_length=512)
        labels = tokenizer(examples['labels'], padding="max_length", truncation=True, max_length=512)
        inputs['labels'] = labels['input_ids']
        return inputs

    # Tokenize the dataset for regular training
    chunked_dataset_regular = chunked_dataset_regular.map(tokenize_function_regular, batched=True)
    chunked_dataset_regular.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    # Define training arguments for regular training
    training_args_regular = TrainingArguments(
        output_dir="./results_regular",
        eval_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=num_epochs,
        logging_dir="./logs_regular",
        logging_steps=10,
        report_to="wandb",
        gradient_checkpointing=True,
        fp16=torch.cuda.is_available(),
    )

    # Initialize and start the Trainer for regular training
    trainer_regular = Trainer(
        model=model, # Use the single loaded model
        args=training_args_regular,
        train_dataset=chunked_dataset_regular,
        eval_dataset=chunked_dataset_regular,
        tokenizer=tokenizer, # Use the single loaded tokenizer
    )

    print(f"Starting Regular training on {device}...")
    trainer_regular.train()
    print("Regular training finished.")

    # --- Prompt Engineering Training (using T5) ---
    print("\n--- Starting Prompt Engineering Training ---")

    # Convert the DataFrame to a Hugging Face Dataset for prompt engineering training (re-sample/slice)
    chunked_dataset_prompt = Dataset.from_pandas(merged_books_df[['input', 'labels']])
    chunked_dataset_prompt = chunked_dataset_prompt.select(range(num_samples))

    # Tokenization function for prompt engineering training (using T5 tokenizer)
    # This can be the same as regular training if using T5 for text-to-text
    def tokenize_function_prompt(examples):
        inputs = tokenizer(examples['input'], padding="max_length", truncation=True, max_length=512)
        labels = tokenizer(examples['labels'], padding="max_length", truncation=True, max_length=512)
        inputs['labels'] = labels['input_ids']
        return inputs

    # Tokenize the dataset for prompt engineering training
    chunked_dataset_prompt = chunked_dataset_prompt.map(tokenize_function_prompt, batched=True)
    chunked_dataset_prompt.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    # Define training arguments for prompt engineering training
    training_args_prompt = TrainingArguments(
        output_dir="./results_prompt",
        eval_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=num_epochs,
        logging_dir="./logs_prompt",
        logging_steps=10,
        report_to="wandb",
        gradient_checkpointing=True,
        fp16=torch.cuda.is_available(),
    )

    # Initialize and start the Trainer for prompt engineering training
    trainer_prompt = Trainer(
        model=model, # Use the single loaded model
        args=training_args_prompt,
        train_dataset=chunked_dataset_prompt,
        eval_dataset=chunked_dataset_prompt,
        tokenizer=tokenizer, # Use the single loaded tokenizer
    )

    print(f"Starting Prompt Engineering training on {device}...")
    trainer_prompt.train()
    print("Prompt Engineering training finished.")

    # --- Model-Agnostic Fine-Tuning (Text-to-Text using T5) ---
    print("\n--- Starting Text-to-Text Fine-Tuning ---")

    # Convert the DataFrame to a Hugging Face Dataset for text-to-text fine-tuning (re-sample/slice)
    chunked_dataset_text2text = Dataset.from_pandas(merged_books_df[['input', 'labels']])
    chunked_dataset_text2text = chunked_dataset_text2text.select(range(num_samples))

    # Tokenization function for text-to-text fine-tuning (using T5 tokenizer)
    # This can be the same as regular and prompt engineering training if using T5 for text-to-text
    def tokenize_function_text2text(examples):
        inputs = tokenizer(examples['input'], padding="max_length", truncation=True, max_length=512)
        labels = tokenizer(examples['labels'], padding="max_length", truncation=True, max_length=512)
        inputs['labels'] = labels['input_ids']
        return inputs

    # Tokenize the dataset for text-to-text fine-tuning
    chunked_dataset_text2text = chunked_dataset_text2text.map(tokenize_function_text2text, batched=True)
    chunked_dataset_text2text.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    # Define training arguments for text-to-text fine-tuning
    training_args_text2text = TrainingArguments(
        output_dir="./results_text2text",
        eval_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=num_epochs,
        logging_dir="./logs_text2text",
        logging_steps=10,
        report_to="wandb",
        gradient_checkpointing=True,
        fp16=torch.cuda.is_available(),
    )

    # Initialize and start the Trainer for text-to-text fine-tuning
    trainer_text2text = Trainer(
        model=model, # Use the single loaded model
        args=training_args_text2text,
        train_dataset=chunked_dataset_text2text,
        eval_dataset=chunked_dataset_text2text,
        tokenizer=tokenizer, # Use the single loaded tokenizer
    )

    print(f"Starting Text-to-Text Fine-Tuning on {device}...")
    trainer_text2text.train()
    print("Text-to-Text Fine-Tuning finished.")

else:
    print("Sequential training skipped.")

print("\n--- Sequential training process completed ---")
