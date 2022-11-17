import torch
import numpy as np
from PIL import Image

model = torch.hub.load('ultralytics/yolov5', 'custom' , path='package/ai_cup_yolo/best.pt')

def predict(path):
  result = model(path)
  return result

def predict_save(image):
  image = Image.open(image.stream)
  image = np.array(image)
  image = predict(image)
  image = image.render()[0]
  image = Image.fromarray(image)
  image = image.resize((960, 540))
  image.save('static/myimg.png')