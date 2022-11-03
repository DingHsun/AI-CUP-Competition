from flask import Flask, redirect, url_for, render_template, request, jsonify
import torch
import numpy as np
import base64
import cv2
from PIL import Image

model = torch.hub.load('ultralytics/yolov5', 'custom' , path='best.pt')
def predict(path):
  result = model(path)
  return result

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
  return render_template('home.html')
@app.route('/index', methods=['GET', 'POST'])
def result():
  if request.method == 'POST':
      image = request.files["img"]
      data = image.filename
      if(data==''):
        return render_template('home.html')
      image = Image.open(image.stream)
      result = np.array(image)
      result = predict(result)
      result = result.render()[0]
      result = Image.fromarray(result)
      result = result.resize((960, 540))
      result.save('static/myimg.png')
      return render_template('result.html')
      #result = np.array(result)
      #result = base64.b64encode(result).decode('utf-8')
      #return render_template('result.html', result = result)
if __name__ == '__main__':
  app.run()