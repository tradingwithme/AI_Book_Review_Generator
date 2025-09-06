import gc
import re
import nltk
import torch
from os import path
import pandas as pd
from json import load
import concurrent.futures
from datasets import Dataset
from accelerate import Accelerator
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain
from alt_book_generator import BookReviewGenerator
from langchain.memory import ConversationBufferMemory
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling

# Preprocessing Class
class Preprocessing:
    def __init__(self, model_name: str, model_max_length: int, summarizer_model_max_length: int):
        self.model_name = model_name
        self.model_max_length = model_max_length
        self.summarizer_model_max_length = summarizer_model_max_length
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.summarizer_model = AutoModelForSeq2SeqLM.from_pretrained("sshleifer/distilbart-cnn-12-6")
        self.summarizer_tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")

    def sanitize_text(self, string_text: str) -> str:
        return re.sub(r'[^A-Za-z0-9\s]+', '', string_text).strip() if isinstance(string_text, str) else ''

    def preprocess_data(self, book_summaries, book_reviews):
        book_content_df = pd.DataFrame(book_summaries, columns=['bookTitle', 'bookContent'])
        book_reviews_df = pd.DataFrame(book_reviews, columns=['bookTitle', 'reviews'])

        # Clean titles and filter out short content
        book_content_df['bookTitle'] = book_content_df['bookTitle'].apply(self.sanitize_text)
        book_reviews_df['bookTitle'] = book_reviews_df['bookTitle'].apply(self.sanitize_text)
        book_content_df = book_content_df[book_content_df['bookContent'].apply(lambda x: len(str(x).split()) > 10)].reset_index(drop=True)
        
        merged_df = pd.merge(book_reviews_df, book_content_df, on='bookTitle', how='inner')
        
        merged_df['combined_text'] = merged_df.apply(lambda row: f"Book Title: {row['bookTitle']}\nBook Content: {row['bookContent']}\nReview: {row['reviews']}", axis=1)
        dataset = Dataset.from_pandas(merged_df)
        
        def tokenize_function(examples):
            return self.tokenizer(examples["combined_text"], truncation=True, max_length=self.model_max_length)

        tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=['bookTitle', 'reviews', 'bookContent', 'combined_text'])
        return tokenized_dataset

    def summarize_review_manual(self, row):
        book_title = row['bookTitle']
        review_text = row['reviews']
        max_review_length = self.model_max_length - len(f"Generate a review for the book: {book_title}")
        summarizer_max_input_length = self.summarizer_model_max_length

        if len(review_text) > max_review_length:
            sentences = nltk.sent_tokenize(review_text)
            summarized_chunks = []
            current_chunk = ""

            for sentence in sentences:
                if len(self.summarizer_tokenizer.encode(current_chunk + sentence)) < summarizer_max_input_length - 5:
                    current_chunk += (sentence + " ")
                else:
                    inputs = self.summarizer_tokenizer(current_chunk, return_tensors="pt", max_length=summarizer_max_input_length, truncation=False).to("cpu")
                    summary_ids = self.summarizer_model.generate(inputs["input_ids"], max_length=summarizer_max_input_length, min_length=30, num_beams=4, early_stopping=True)
                    summarized_chunk = self.summarizer_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                    summarized_chunks.append(summarized_chunk)
                    current_chunk = (sentence + " ")

            if current_chunk:
                inputs = self.summarizer_tokenizer(current_chunk, return_tensors="pt", max_length=summarizer_max_input_length, truncation=False).to("cpu")
                summary_ids = self.summarizer_model.generate(inputs["input_ids"], max_length=summarizer_max_input_length, min_length=30, num_beams=4, early_stopping=True)
                summarized_chunk = self.summarizer_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                summarized_chunks.append(summarized_chunk)

            summarized_text = " ".join(summarized_chunks)

            while len(self.summarizer_tokenizer.encode(summarized_text)) > max_review_length:
                inputs = self.summarizer_tokenizer(summarized_text, return_tensors="pt", max_length=summarizer_max_input_length, truncation=False).to("cpu")
                summary_ids = self.summarizer_model.generate(inputs["input_ids"], max_length=summarizer_max_input_length, min_length=30, num_beams=4, early_stopping=True)
                summarized_text = self.summarizer_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

            return summarized_text
        else:
            return review_text

# Training Class
class Training:
    def __init__(self, model_name, model_max_length, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, dataset: Dataset):
        self.model = model
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.model_max_length = model_max_length
        
        try:
            import torch_xla.core.xla_model as xm
            self._has_tpu = True
        except ImportError:
            self._has_tpu = False

        # Set the device to TPU if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Adjust training batch size based on device
        if self.device == "cuda":
            self.train_batch_size = 8
            self.eval_batch_size = 8
        else:
            self.train_batch_size = 1
            self.eval_batch_size = 1

    def fine_tune(self, output_dir: str, epochs: int = 3, batch_size: int = 4, learning_rate: float = 5e-5):
        train_test_split = self.dataset.train_test_split(test_size=0.2)
        train_dataset = train_test_split['train']
        eval_dataset = train_test_split['test']
        accelerator = Accelerator(mixed_precision="no")
        training_args = TrainingArguments(
output_dir="./output/gpt-neo-fine-tuned",
overwrite_output_dir=True,
num_train_epochs=epochs,
per_device_train_batch_size=self.train_batch_size,
per_device_eval_batch_size=self.eval_batch_size,
learning_rate=5e-5,
weight_decay=0.01,
eval_strategy="epoch",
save_strategy="epoch",
load_best_model_at_end=True,
metric_for_best_model="eval_loss",
logging_dir="./logs",
logging_steps=10,
report_to="none",
fp16=False,  # Disable FP16 mixed precision
bf16=False,  # Disable BF16 mixed precision
)
        if self.device == 'cuda':
            print(f"Detected device: {self.device}. Attempting to use Unsloth.")
            # Import Unsloth components
            try:
                import unsloth
                from unsloth import FastLanguageModel
                if 'FastTrainer' in dir(unsloth): 
                    from unsloth import FastTrainer
                    Trainer_class = FastTrainer
                elif 'UnslothTrainer' in dir(unsloth):
                    from unsloth import UnslothTrainer
                    Trainer_class = UnslothTrainer
                else: 
                    Trainer_class = Trainer
                # Wrap the model with Unsloth
                # You might need to adjust max_seq_length based on your tokenized data if not using max_length=model_max_length
                try: 
                    model = FastLanguageModel.from_pretrained(
model_name = self.model_name, # Use the original model name
max_seq_length = self.model_max_length,
dtype = None, # None for auto detection
load_in_4bit = True, # Load in 4bit for memory efficiency
)
                    print("Model wrapped with Unsloth.")
                except ModuleNotFoundError: pass
            
            except ImportError:
                print("Unsloth not installed. Proceeding without Unsloth.")
                # If Unsloth is not installed, use the standard Trainer
                Trainer_class = Trainer
                print("Using standard Trainer.")

            except Exception as e:
                print(f"Error during Unsloth setup: {e}. Proceeding with standard Trainer.")
                # If there's an error with Unsloth setup, use the standard Trainer
                Trainer_class = Trainer
                print("Using standard Trainer.")
    
        else:
            print(f"Detected device: {self.device}. Proceeding without Unsloth.")
            # Use the standard Trainer on CPU
            Trainer_class = Trainer
            print("Using standard Trainer.")
                
        trainer = Trainer_class(
model=self.model.to(self.device),
args=training_args,
train_dataset=train_dataset,
eval_dataset=eval_dataset,
data_collator=DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False),
)
        if self.device != "cuda": trainer = accelerator.prepare(trainer)
        trainer.train()
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)

# Review Generation Class
class ReviewGeneration:
    def __init__(self, model_name: str, model_max_length: int, tokenizer: AutoTokenizer):
        self.model_name = model_name
        self.model_max_length = model_max_length
        self.tokenizer = tokenizer
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        
        self.prompt_template = """You are an AI assistant helping to generate a book review based on the given book title, book content, and previous reviews.
Generate a thoughtful and insightful review for the following book:
Book Title: {book_title}
Book Content: {book_content}
Previous Reviews: {previous_reviews}
"""
        self.prompt = PromptTemplate(input_variables=["book_title", "book_content", "previous_reviews"], template=self.prompt_template)
        
        self.conversation_chain = ConversationChain(
            memory=self.memory,
            prompt=self.prompt,
            llm=AutoModelForCausalLM.from_pretrained(model_name),
            verbose=True
        )
    
    def generate_review(self, book_title: str, book_content: str, previous_reviews: list):
        book_title_tokens = self.tokenizer.encode(book_title, truncation=True, max_length=self.model_max_length)
        book_content_tokens = self.tokenizer.encode(book_content, truncation=True, max_length=self.model_max_length)
        previous_reviews_tokens = self.tokenizer.encode(previous_reviews, truncation=True, max_length=self.model_max_length)

        # Run the model for generating the review based on the tokens
        response = self.conversation_chain.run(book_title=book_title, book_content=book_content, previous_reviews=previous_reviews)
        return response

def fine_tune_model(book_title: str,
book_content: str, 
book_reviews: list,
fine_tune: bool = True,
generate: bool = True):
    # Model Setup for Main Model (GPT Neo) and Summarization (BART)
    summarizer_model_name = "sshleifer/distilbart-cnn-12-6"

    # Load the models and tokenizers
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    summarizer_model = AutoModelForSeq2SeqLM.from_pretrained(summarizer_model_name)
    summarizer_tokenizer = AutoTokenizer.from_pretrained(summarizer_model_name)

    # Model max lengths
    model_max_length = model.config.max_position_embeddings
    summarizer_model_max_length = summarizer_model.config.max_position_embeddings
    tokenizer.pad_token = tokenizer.eos_token  # Set padding token for GPT

    # Device setup for CPU or GPU
    device = torch.device("cpu")  # Default to CPU
    model_name = "gpt-neo-fine-tuned" if path.exists("./gpt-neo-fine-tuned") else "EleutherAI/gpt-neo-125M"
    preprocessor = Preprocessing(model_name, model_max_length, summarizer_model_max_length)

    if model_name == "EleutherAI/gpt-neo-125M" or fine_tune:
        with open('bookSummaries.json','r',encoding='utf-16') as file: book_summaries = load(file)
        with open('bookReviews.json','r',encoding='utf-16') as file: book_reviews2 = load(file )
        tokenized_data = preprocessor.preprocess_data(book_summaries, book_reviews2)
        trainer = Training(model_name, model_max_length, model, tokenizer, tokenized_data)
        trainer.fine_tune("./gpt-neo-fine-tuned")
        del trainer, tokenized_data
        gc.collect()
    book_summaries = [(book_title, book_content)]
    tokenized_data = preprocessor.preprocess_data(book_summaries, book_reviews)
    trainer = Training("gpt-neo-fine-tuned", model_max_length, model, tokenizer, tokenized_data)
    trainer.fine_tune("./gpt-neo-fine-tuned")
    if generate:
        review_generator = ReviewGeneration(model_name, model_max_length, tokenizer)
        gc.collect()
        review = review_generator.generate_review(book_title, book_content, book_reviews)
        #print(f"Generated Review: {review}")
        del review_generator
    del trainer, tokenized_data
    gc.collect()
    if generate: return review

def fine_tune_and_generate(book_title, book_content, book_reviews):
    # Run both functions simultaneously using threading
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_fine_tune = executor.submit(fine_tune_model, book_title, book_content, book_reviews, generate=False)
        future_generate = executor.submit(BookReviewGenerator, book_title, book_content, book_reviews)

        # Wait for both operations to finish and get the results
        fine_tune_review = future_fine_tune.result()
        generated_review = future_generate.result()

        print(f"Fine-tuned Review: {fine_tune_review}")
        print(f"Generated Review: {generated_review}")
        
    return fine_tune_review, generated_review