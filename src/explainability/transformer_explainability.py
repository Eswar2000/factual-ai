import torch
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import BertTokenizer, BertModel

# Load pretrained model (same as fine-tuned if available)
tokenizer = BertTokenizer.from_pretrained("data/models/transformer/final_model")
model = BertModel.from_pretrained("data/models/transformer/final_model", output_attentions=True)
model.eval()

# Your example input
sentence = "The claim about healthcare is misleading."
inputs = tokenizer(sentence, return_tensors="pt")
outputs = model(**inputs)

# Get attention from last layer, first head
attention = outputs.attentions[-1][0][0].detach().numpy()  # shape: (seq_len, seq_len)
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

# Plot
plt.figure(figsize=(10, 8))
sns.heatmap(attention, xticklabels=tokens, yticklabels=tokens, cmap="Blues", annot=False)
plt.title("BERT Attention Heatmap (Last Layer, Head 0)")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("data/explainability/bertwiz/transformer_attention_sample_0.png")
plt.close()

print("✅ Saved transformer attention heatmap.")
