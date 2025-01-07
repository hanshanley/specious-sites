import torch
import gc
from types import SimpleNamespace
from transformers import AutoTokenizer
from contrastive_base import ContrastiveModel
import torch.nn.functional as F

# Check and configure the device
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    torch.cuda.empty_cache()
else:
    device = torch.device('cpu')
    print("Using CPU")

# Configuration for the Contrastive Model
MODEL = 'intfloat/e5-base-v2'
config = {
    'hidden_dropout_prob': 0.1,
    'model_name': MODEL,
    'attention_probs_dropout_prob': 0.1,
    'hidden_size': 768,
}
config = SimpleNamespace(**config)

# Initialize the Contrastive Model
model = ContrastiveModel(config)

model = ContrastiveModel(config)
saved = torch.load('/mnt/projects/qanon_proj/ContrastiveMisinformationModel/pretrain-1-3e-05-contrastive_model-20230721.pt')
model.load_state_dict(saved['model'])
model = model.to(device)
model.eval()

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL)

# Helper function for mean pooling
def mean_pooling(model_output, attention_mask):
    """
    Perform mean pooling on the token embeddings.
    Args:
        model_output: The output from the transformer model.
        attention_mask: The attention mask to exclude padding tokens.
    Returns:
        Pooled sentence embeddings.
    """
    token_embeddings = model_output[0]  # First element contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# Function to get sentence embeddings
def get_sentence_embeddings(model, sentences):
    """
    Compute embeddings for a list of sentences using the specified model.
    Args:
        model: The model to generate embeddings.
        sentences: List of input sentences.
    Returns:
        List of normalized sentence embeddings.
    """
    all_embeddings = []
    num_batches = len(sentences) // 32 + (1 if len(sentences) % 32 > 0 else 0)

    for batch in range(num_batches):
        # Tokenize the batch of sentences
        encoded_input = tokenizer(
            sentences[batch * 32:(batch + 1) * 32],
            max_length=256,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )

        with torch.no_grad():
            # Get model output and compute sentence embeddings
            model_output = model(
                encoded_input['input_ids'].to(device),
                encoded_input['attention_mask'].to(device)
            )
            sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'].to(device))
            sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        
        # Store embeddings
        all_embeddings.extend(sentence_embeddings.cpu().numpy())
    
    return all_embeddings
