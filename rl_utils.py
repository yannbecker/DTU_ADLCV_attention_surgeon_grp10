import torch 

"""
REPENSER A CA, comment recuperer et pruning 

"""
num_heads = 12
head_dim = 768 // num_heads

def split_qkv_heads(model, i):
    """
    Dinov2 store 12 qkv matrices in one tensor. This function is a helper to go from (num_heads * head_dim * 3_qkv , out_features) -> (num_heads, head_dim, 3_qkv , out_features).
    Operate for the i-th block.
    """
    qkvs = model.blocks[i].attn.qkv.weight.reshape(num_heads, 3, head_dim, 768)
    return qkvs

def get_head_attention(model, x, i):
    """
    Get the attention_map for input x and block i
    """
    B, N, C = x.shape
    qkv = model.blocks[i].attn.qkv(x).reshape(B, N, 3, num_heads, C // num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]

    # Calcul de la matrice d'attention : (Softmax(QK^T / sqrt(d)))
    attn = (q @ k.transpose(-2, -1)) * (head_dim ** -0.5)
    attn = attn.softmax(dim=-1) 
    
    return attn