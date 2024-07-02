import torch


#### the gradient is calculated based on the
inp = torch.randint(low=0, high=10, size=(2, 3), dtype=torch.float32, requires_grad=True)
out = (inp+1).pow(2)
out.backward(torch.ones_like(out), retain_graph=True)
print(f"input\n{inp}")
print(f"output\n{out}")
print(f"First call\n{inp.grad}")
out.backward(torch.ones_like(out), retain_graph=True)
print(f"\nSecond call\n{inp.grad}")
inp.grad.zero_()
out.backward(torch.ones_like(out), retain_graph=True)
print(f"\nCall after zeroing gradients\n{inp.grad}")




