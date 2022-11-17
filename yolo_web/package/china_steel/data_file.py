import torch
from torch.utils.data import DataLoader

char = [" ", "0", "1", "2", "3", "4", "5", "6", "7", "8",
     "9", "A", "B", "C", "D", "E", "F", "G", "H", "I",
     "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T",
     "U", "V", "W", "X", "Y", "Z"]
char = {stu:i for i,stu in zip(range(len(char)),char)}
def embedding(label):
  embed = torch.zeros(16)
  j = 2
  for i in range(len(label)):
    if char[label[i]] == char[label[i-1]]:
      j += 1
    embed[j] = char[label[i]]
    j += 1
  return embed

def un_embedding(out):
  unembed = ""
  n = len(list(out))
  for i in range(n):
    if (out[i] != 0) & (out[i] != out[i-1]):
      unembed = unembed + list(char.keys())[list(char.values()).index(out[i])]
  return unembed

class predict_data(torch.utils.data.Dataset):
  def __init__(self, data):
    self.datalist = []
    self.datalist.append(data)

  def __getitem__(self, index):
    image = self.datalist[index]
    return image

  def __len__(self):
    return len(self.datalist)

def predict_dataloader(data):
  dataset_test = predict_data(data)
  test_loader = DataLoader(dataset_test, batch_size=100, shuffle=False, drop_last=False)
  return test_loader

from numpy import newaxis
from torchvision import transforms
import package.china_steel.model_p as model_p
from package.china_steel.model_p import MY_Model
model = model_p.point_model_load(iscuda=False)
c1, c2, c3, c4 = 450, 550, 320, 900
cut_range = 70
d = 8
transforms_point = transforms.Compose([
  transforms.Resize((int((c2-c1+2*cut_range)/d),int((c4-c3+2*cut_range)/d))),
  #transforms.ToTensor()
])
transform_resize = transforms.Compose([
  #transforms.ToPILImage(),
  transforms.Resize((32,160)),
  #transforms.ToTensor()
])
def testing_data(path, transform=transform_resize, c=(70, 450, 550, 320, 900, 8), transforms_point=transforms_point):
  data = path
  data = data[newaxis,:,:]
  data = torch.FloatTensor(data)
  cut_range, c1, c2, c3, c4, d = c
  data_ = data[:,c1-cut_range:c2+cut_range,c3-cut_range:c4+cut_range]
  data_ = transforms_point(data_)
  point = model(data_.resize(1,1,30,90)).resize(4)
  o1 = point[0]*d - cut_range + c3
  o2 = point[1]*d - cut_range + c1
  o3 = point[2]*d - cut_range + c3
  o4 = point[3]*d - cut_range + c1
  data = data[:,o2.int():o4.int(), o1.int():o3.int()]
  data = transform(data)
  return data