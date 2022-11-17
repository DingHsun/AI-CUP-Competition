import torch
import package.china_steel.data_file as data_file
import package.china_steel.model_resnet as model_resnet

model = model_resnet.resnet_model_load(iscuda=False)

def predict(input_data):
  output_list = []
  with torch.no_grad():
    for data in input_data:
      images = data ##.cuda()
      outputs = model(images)
      outputs = outputs.argmax(2)
      outputs = outputs.permute(1,0)
      for i in range(outputs.size(0)):
        output_list.append(data_file.un_embedding(outputs[i]))
  return output_list