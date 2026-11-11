import torch
sync=getattr(torch.cuda,'synchronize')
sdpa=getattr(torch.nn.attention,'sdpa_kernel',None)
mm=getattr(torch.ops.aten,'mm')
matmul=getattr(torch.backends.cuda,'matmul')
