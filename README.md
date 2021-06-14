# locoprop-pytorch [WIP]
Unofficial PyTorch implementation of LocoProp, an enhanced BackProp alternative.

### Preface
'LocoProp: Enhanced BackProp via Local Loss Optimization' by Ehsan Amid, Rohan Anil, and Manfred K. Warmuth. [[`abs`](https://arxiv.org/abs/2106.06199), [`pdf`](https://arxiv.org/pdf/2106.06199.pdf)] introduces a fresh approach to typical learning via Back Propagation.

### Installation
You can install the `locoprop-pytorch` package via `pip`:

```bash
pip install locoprop-pytorch
```

### Usage
You're probably familiar with defining the `forward` method in your `nn.Module` models. You can also define custom backward passes using the `torch.autograd` module. Writing a `backward` method enables you to have full control over how you want  back propagation to be done in your networks.

You can use the `LocoProp` algorithm like so:

```python
import torch
from torch import nn
from locoprop_pytorch import LocoProp

class MyNet(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.layers = nn.ModuleList([...])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x

    # TODO: finish backward example code
    def backward(self):
        pass
```