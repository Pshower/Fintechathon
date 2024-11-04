import numpy as np
# pip install numpy==1.22.4
import torch
print(torch.__version__) # 2.4.1
print(np.__version__) # 1.19.2
fmax = 10 
d = torch.Tensor([1.,2.,3.,4.])
c = np.array([1,2,3,4])
print(c)
print(np.sinc(c))
print(d)
e = d.numpy()
print(e)
print("np.sinc(d.numpy()):",np.sinc(e))
