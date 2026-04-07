"""
This code plot the activation of each head for a specific task/dataset. This takes place before the RL pruning.
Pipeline ideation: 
Select task/dataset -> for each image/batch: -> Forward the input and compute the accuracy
                                             -> Get the Attention map of this input for each of the 144 heads (in a tensor attn_heads of shape (12*12))
                    -> Average and compute the 4 metrics 
                    -> plot the heatmap 
"""

