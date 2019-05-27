from pprint import pprint as pp

import torch
import torch.nn as nn

m = nn.Sequential(
    nn.Conv2d(3, 2, 3, bias=False, padding=1),
    nn.BatchNorm2d(2),
    nn.LeakyReLU(0.1)
)

m.eval()

conv_weight = m[0].weight.type(torch.double)
bn_weight = m[1].weight.type(torch.double)

image_size = 416

inputs = torch.randn([1, 3, image_size, image_size])
inputs.requires_grad = True
outputs = m(inputs)

outputs.mean().backward()
bp_gradient = inputs.grad
pp(bp_gradient)
pp(bn_weight)
pp(conv_weight)

weights = torch.stack([torch.stack([conv_weight for _ in range(image_size)], 1) for _ in range(image_size)], 1)
weights = torch.unsqueeze(weights, 0)

grads_manual = torch.zeros_like(inputs)
padder = torch.nn.ConstantPad2d(1, 0)

leaky_output_gradient = (outputs > 0).type(torch.double) * 0.9 + 0.1
padded = padder(grads_manual.clone()).type(torch.double)

bned_leaky = torch.einsum("ijkl, j->ijkl", leaky_output_gradient, bn_weight)
bned_leaky_weight = torch.einsum("ijkl, ijklmno->ijklmno", bned_leaky, weights.type(torch.double))

leaky_mean_weight = bned_leaky_weight.mean(1).permute(0, 3, 1, 2, 4, 5)

for i in range(3):
    for j in range(3):
        padded[:, :, i:image_size + i, j:image_size + j] += leaky_mean_weight[..., i, j]

grads_manual = padded[:, :, 1:-1, 1:-1] / image_size / image_size

pp(grads_manual)
pp(bp_gradient)
