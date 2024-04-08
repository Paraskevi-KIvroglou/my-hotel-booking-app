from transformers import AutoModelForCausalLM, AutoTokenizer
import os

def save_model_and_tokenizer_to_cache(model_id, cache_dir):
    # Load tokenizer with local caching
    if not os.path.exists(cache_dir):
        tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)

        # Load model with local caching
        model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=cache_dir)

        # Save tokenizer locally
        tokenizer.save_pretrained(cache_dir)

        # Save model locally
        model.save_pretrained(cache_dir)

def load_model(cached_path):
    tokenizer = AutoTokenizer.from_pretrained(cached_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(cached_path, local_files_only=True)
    
    return tokenizer,model

model_id = "KvrParaskevi/Hotel-Assistant-Attempt5-Llama-2-7b"  # Replace with your model ID
cache_dir = "cache_memory"  # Replace with the directory where you want to save the models and tokenizers

save_model_and_tokenizer_to_cache(model_id, cache_dir)