import torch

# The released models were trained and evaluated with TF32 matmul enabled.
torch.backends.cuda.matmul.allow_tf32 = True
