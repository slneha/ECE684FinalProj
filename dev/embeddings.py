"""
Generate embeddings for tweets using Qwen3 embedding model.
Creates tweet-level embeddings at 1024 dimensions and outputs to JSON.
"""

import pandas as pd
import json
import torch
import os

# Check transformers version and provide helpful error message if needed
try:
    from transformers import AutoTokenizer, AutoModel
    import transformers
    print(f"Using transformers version: {transformers.__version__}")
except ImportError:
    print("ERROR: transformers library not installed.")
    print("Please install it with: pip install transformers torch")
    raise

# Configuration
CSV_FILE = "Tweets_clean.csv"
OUTPUT_FILE = "tweet_embeddings.json"
MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"  # Qwen3 embedding model
EMBEDDING_DIM = 1024
BATCH_SIZE = 32

# Instruction for Qwen3 embeddings (instruction-aware model)
EMBEDDING_INSTRUCTION = "Represent the following tweet for retrieval:"

def load_model():
    """Load the Qwen3 embedding model and tokenizer."""
    print(f"Loading model: {MODEL_NAME}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)
    except ValueError as e:
        if "model type" in str(e) or "does not recognize" in str(e):
            print("\n" + "="*60)
            print("ERROR: Transformers library doesn't recognize Qwen3 model type.")
            print("Please upgrade transformers to the latest version:")
            print("  pip install --upgrade transformers")
            print("\nOr install from source (for latest Qwen3 support):")
            print("  pip install git+https://github.com/huggingface/transformers.git")
            print("="*60)
            raise
        else:
            raise
    
    # Set model to evaluation mode
    model.eval()
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Model loaded on device: {device}")
    
    return tokenizer, model, device

def generate_embeddings(texts, tokenizer, model, device):
    """Generate embeddings for a batch of texts using Qwen3."""
    # Prepare texts with instruction (Qwen3 is instruction-aware)
    instruction_texts = [f"{EMBEDDING_INSTRUCTION} {text}" for text in texts]
    
    # Tokenize
    encoded = tokenizer(
        instruction_texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    
    # Move to device
    encoded = {k: v.to(device) for k, v in encoded.items()}
    
    # Generate embeddings
    with torch.no_grad():
        outputs = model(**encoded)
        last_hidden_states = outputs.last_hidden_state
        
        # Get attention mask to ignore padding tokens
        attention_mask = encoded["attention_mask"]
        
        # Use mean pooling: average all token embeddings (excluding padding)
        # Expand attention mask to match hidden state dimensions
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_states.size()).float()
        
        # Sum embeddings, masking out padding
        sum_embeddings = torch.sum(last_hidden_states * mask_expanded, dim=1)
        
        # Count non-padding tokens
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        
        # Compute mean
        embeddings = sum_embeddings / sum_mask
    
    # Normalize embeddings (L2 normalization)
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    
    return embeddings.cpu().numpy()

def main():
    """Main function to generate embeddings."""
    # Load data
    csv_path = os.path.join(os.path.dirname(__file__), "..", CSV_FILE)
    print(f"Loading data from {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Use clean_text column for embeddings
    texts = df["clean_text"].fillna("").tolist()
    tweet_ids = df["tweet_id"].tolist()
    
    print(f"Loaded {len(texts)} tweets")
    
    # Load model
    tokenizer, model, device = load_model()
    
    # Generate embeddings in batches
    all_embeddings = []
    print("Generating embeddings...")
    
    for i in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[i:i + BATCH_SIZE]
        batch_embeddings = generate_embeddings(batch_texts, tokenizer, model, device)
        all_embeddings.extend(batch_embeddings.tolist())
        
        if (i + BATCH_SIZE) % 100 == 0 or (i + BATCH_SIZE) >= len(texts):
            print(f"Processed {min(i + BATCH_SIZE, len(texts))} / {len(texts)} tweets")
    
    print(f"Generated embeddings for {len(all_embeddings)} tweets")
    print(f"Embedding dimension: {len(all_embeddings[0])}")
    
    # Prepare output data
    output_data = {
        "embeddings": {
            str(tweet_id): embedding 
            for tweet_id, embedding in zip(tweet_ids, all_embeddings)
        },
        "metadata": {
            "model": MODEL_NAME,
            "dimension": len(all_embeddings[0]),
            "num_tweets": len(tweet_ids),
            "instruction": EMBEDDING_INSTRUCTION
        }
    }
    
    # Save to JSON
    output_path = os.path.join(os.path.dirname(__file__), "..", OUTPUT_FILE)
    print(f"Saving embeddings to {output_path}")
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    
    print("Done!")

if __name__ == "__main__":
    main()

