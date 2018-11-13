# with open("../../data/corpus/cover_implicit.label") as f:
#     lines = f.readlines()
#     zero = 0
#     one = 0
#     for line in lines:
#         line = line.strip()
#         if line == "0":
#             zero += 1
#         if line == "1":
#             one += 1
#     print(len(lines))
#     print("0:",zero)
#     print("1:",one)
import torch
a = torch.Tensor([[1,2],[3,4]])
b,c = torch.max(a,dim=1)
print(b,c)
