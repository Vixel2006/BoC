import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_concept_alignment(img_indices, txt_indices, batch_idx=0):
    # Take the first example in the batch
    img_codes = img_indices[batch_idx].cpu().numpy() # e.g., [5, 42, 12, 5, 0]
    txt_codes = txt_indices[batch_idx].cpu().numpy() # e.g., [42, 5, 9, 0, 0]
    
    # Create a correlation matrix
    # Rows = Image Slots, Cols = Text Slots
    # We will fill it with the Concept ID if they match, else -1
    n_img = len(img_codes)
    n_txt = len(txt_codes)
    matrix = np.full((n_img, n_txt), -1)
    
    for i in range(n_img):
        for j in range(n_txt):
            if img_codes[i] == txt_codes[j]:
                matrix[i, j] = img_codes[i] # Store the Concept ID (e.g., 42)

    # Plot
    fig, ax = plt.subplots()
    # Use a discrete colormap (like 'tab20') so every concept ID has a distinct color
    cmap = plt.get_cmap('tab20') 
    
    # We use a masked array to hide non-matches (make them white/black)
    masked_matrix = np.ma.masked_where(matrix == -1, matrix)
    
    im = ax.imshow(masked_matrix, cmap=cmap, vmin=0, vmax=64) # Assuming 64 concepts
    
    # Labels
    ax.set_ylabel("Image Slots")
    ax.set_xlabel("Text Slots")
    ax.set_title(f"Concept Alignment (Sample {batch_idx})")
    
    # Annotate the grid with the actual Concept IDs
    for i in range(n_img):
        for j in range(n_txt):
            val = matrix[i, j]
            if val != -1:
                text = ax.text(j, i, str(val),
                               ha="center", va="center", color="black", weight="bold")
                
    return fig

def plot_attention_map(attention_weights, x_labels=None, y_labels=None, title="Attention Map"):
    if isinstance(attention_weights, torch.Tensor):
        attention_weights = attention_weights.detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(attention_weights, cmap='viridis', origin='upper')

    # We want to show all ticks...
    ax.set_xticks(np.arange(attention_weights.shape[1]))
    ax.set_yticks(np.arange(attention_weights.shape[0]))
    # ... and label them with the respective list entries.
    if x_labels is not None:
        ax.set_xticklabels(x_labels)
    if y_labels is not None:
        ax.set_yticklabels(y_labels)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    ax.set_title(title)
    fig.tight_layout()
    return fig

def plot_metric_per_epoch(scores, metric_name, title="Metric per Epoch"):
    epochs = range(1, len(scores) + 1)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, scores, marker='o', linestyle='-')
    
    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric_name)
    ax.set_title(title)
    ax.grid(True)
    
    return fig

