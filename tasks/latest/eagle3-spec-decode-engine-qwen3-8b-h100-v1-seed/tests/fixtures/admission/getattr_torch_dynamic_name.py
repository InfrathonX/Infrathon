import torch
def probe(name):
 return getattr(torch.backends.cuda,name)
