import numpy as np
import os
import pickle
import argparse
import matplotlib
import matplotlib.pyplot as plt
import argparse
import os
import spacy


def load_dir(dir):
    output_list=[]
    file_list=os.listdir(dir)
    for filename in file_list:
        if 'png' in filename or 'pkl' in filename:
            continue
        path=os.path.join(dir,filename)
        f=open(path,'r')
        tmp=f.readlines()
        f.close()
        tmp=[i.strip() for i in tmp]
        output='\n'.join(tmp)
        output_list.append(output)
    return output_list

def read_file_list(file_list):
    output_list=[]
    for path in file_list:
        if 'png-' in path:
            pass
        elif '.png' in path or '.pkl' in path:
            continue
        f=open(path,'r')
        tmp=f.readlines()
        f.close()
        tmp=[i.strip() for i in tmp]
        output='\n'.join(tmp)
        output_list.append(output)
    return output_list


def load_file(path):
    f=open(path,'r')
    tmp=f.readlines()
    f.close()
    tmp=[i.strip() for i in tmp]
    return tmp

# Compute KL Divergence based on similarity scores
def kl_divergence(p, q):
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))


def get_similarity(s1, s2, method='spacy', misc=None):
    """
    Compute similarity between strings s1, s2 using either a spaCy model or
    a Hugging Face transformer model.

    :param s1: First string
    :param s2: Second string
    :param method: 'spacy' or 'transformer'
    :param misc:
        - If method='spacy', pass your spaCy model (callable) here, e.g. spacy_model
        - If method='transformer', pass [tokenizer, model] here
    :return: a float similarity score
    """

    if method == 'spacy':
        # Expect 'misc' to be a spaCy model, e.g. nlp = spacy.load("en_core_web_md")
        if not callable(misc):
            raise ValueError("For method='spacy', `misc` must be a spaCy model (callable).")
        doc1 = misc(s1)
        doc2 = misc(s2)
        similarity_score = doc1.similarity(doc2)
        return similarity_score

    elif method == 'transformer':
        # Expect 'misc' to be [tokenizer, model]
        import torch
        from scipy.spatial.distance import cosine
        if not isinstance(misc, (list, tuple)) or len(misc) < 2:
            raise ValueError("For method='transformer', `misc` must be [tokenizer, model].")

        tokenizer, model = misc[0], misc[1]

        def get_embedding(text):
            # Tokenize and encode the text
            inputs = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
            input_ids = inputs['input_ids']

            # Get the embeddings
            with torch.no_grad():
                outputs = model(input_ids)
            # Mean of last hidden state as the sentence representation
            embeddings = outputs.last_hidden_state.mean(dim=1).squeeze()
            return embeddings

        embedding1 = get_embedding(s1)
        embedding2 = get_embedding(s2)
        similarity_score = 1 - cosine(embedding1, embedding2)
        return similarity_score

    else:
        raise ValueError(f"Invalid method='{method}'. Must be 'spacy' or 'transformer'.")


def get_divergence(similarity_matrix,i,j,mode='KL'):
    if mode=='KL':
        # Calculate KL Divergence
        p = similarity_matrix[i] / np.sum(similarity_matrix[i])
        q = similarity_matrix[j] / np.sum(similarity_matrix[j])
        divergence = np.sum(p * np.log(p / q))
    return divergence

def visualize(divergence_matrix,save_path,vmax):
    # Create a heatmap
    plt.figure(figsize=(8, 8))
    norm=matplotlib.colors.Normalize(vmin=0,vmax=vmax)
    plt.imshow(divergence_matrix, cmap='viridis', interpolation='nearest',norm=norm)
    plt.colorbar(label='Divergence')

    plt.title('Divergence Matrix Heatmap')
    plt.xticks(range(divergence_matrix.shape[0]))
    plt.yticks(range(divergence_matrix.shape[0]))

    # Save the heatmap as an image file (e.g., PNG)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

