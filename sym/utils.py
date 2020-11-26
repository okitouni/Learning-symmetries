import torch

@torch.jit.script
def Augment(imgs):
    nums = []
    for img in imgs:
        i = torch.randint(0,10,(1,)).item()
        j = torch.randint(0,10,(1,)).item()
        istart = i * 28
        iend = (i+1) * 28
        jstart = i * 28
        jend = (i+1) * 28
        zeros = torch.zeros(280, 280)
        zeros[istart:iend, jstart:jend] = img
        nums.append(zeros)
    return torch.stack(nums)
