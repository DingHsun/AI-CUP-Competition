import torch
import torch.nn as nn 
import torchvision.models as models

class resnet_Model(nn.Module): 
  def __init__(self): 
    super().__init__()
    self.h_n = torch.zeros(2,100,256)    #(128, 1, 32, 160)
    #input_shape=(1,32,160)
    self.cnn_model = models.resnet50(pretrained=True)
    self.cnn_model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False) #output_shape=(64,32,160)
    self.cnn_model.layer3[0].conv2 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1), bias=False)
    self.cnn_model.layer3[0].downsample[0] =  nn.Conv2d(512, 1024, kernel_size=(1, 1), stride=(2, 1), bias=False)
    self.cnn_model.layer4[0].conv2 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1), bias=False)
    self.cnn_model.layer4[0].downsample[0] = nn.Conv2d(1024, 2048, kernel_size=(1, 1), stride=(2, 1), bias=False)
    self.cnn_model = nn.Sequential(*(list(self.cnn_model.children())[:-2]))
    ct = 0
    for child in self.cnn_model.children():
      ct += 1
      if ct < 5:
        for param in child.parameters():
          param.requires_grad = False
    self.maxpool = nn.MaxPool2d(kernel_size=(2, 1)) #output_shape=(512,1,40)
    #output_shape=(512,1,40)
    #output_shape=(512,40)
    self.gru1 = nn.GRU(input_size=2048, hidden_size=256, bidirectional=True)   #output_shape=(256,40)
    self.gru2 = nn.GRU(input_size=512, hidden_size=256, bidirectional=True)   #output_shape=(256,40)
    self.fc = nn.Linear(512, 36)

  def forward(self, x): 
    if not isinstance(x, torch.Tensor):
      x = torch.Tensor(x)
    out = self.cnn_model(x)
    out = self.maxpool(out)
    out = torch.squeeze(out, dim=2)              #output_shape=(128,512,40)
    out = out.permute(2, 0, 1)                #output_shape=(40,128,512)
    out, _ = self.gru1(out, self.h_n) #output_shape=(40,128,256)
    out, _ = self.gru2(out, self.h_n)           #output_shape=(40,128,256)
    out = self.fc(out)                     #output_shape=(40,128,36)
    return out

def resnet_model_load(iscuda = False):
  if(iscuda):
    return torch.load("package\china_steel\model\picachu_model_gru.pth")
  else:
    return torch.load("package\china_steel\model\picachu_model_gru.pth", map_location=torch.device('cpu'))