import torch
import torch.nn as nn
import torchvision.models as models

class MY_Model(nn.Module): 
  def __init__(self): 
    super().__init__()
    self.res = nn.Sequential(*list(models.resnet18(pretrained=True).children())[:-1])
    self.res[0] = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
    ct = 0
    for child in self.res.children():
      ct += 1
      if ct < 4:
        for param in child.parameters():
          param.requires_grad = False
    res_out = 512
    self.out = nn.Linear(res_out, 4)

  def forward(self, x): 
    if not isinstance(x, torch.Tensor):
      x = torch.Tensor(x)
    x = self.res(x)
    x = x.view(x.size(0),-1)#/255
    #x = torch.cat((x.mean(2),x.mean(3)),2)/100
    out = self.out(x)
    return out


def point_model_load(iscuda = False):
  if(iscuda):
    return torch.load("package\china_steel\model\picachu_model_標點model.pth")
  else:
    return torch.load("package\china_steel\model\picachu_model_標點model.pth", map_location=torch.device('cpu'))