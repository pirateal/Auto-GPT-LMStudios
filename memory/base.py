from transformers import AutoTokenizer, AutoModel
import torch

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('izhx/udever-bloom-1b1')
model = AutoModel.from_pretrained('izhx/udever-bloom-1b1')

def get_ada_embedding(text):
    # Tokenize input text
    inputs = tokenizer(text, return_tensors="pt")

    # Generate embeddings
    with torch.no_grad():  # Disable gradient calculations for inference
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state

    # Process embeddings as needed (e.g., mean pooling)
    # Here, we take the mean of all token embeddings
    return torch.mean(embeddings, dim=1).numpy()




class MemoryProviderSingleton:
    pass
