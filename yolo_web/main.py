from flask import Flask, redirect, url_for, render_template, request, jsonify
from package.ai_cup_yolo.yolo import predict_save
from package.china_steel.model_p import MY_Model
from package.china_steel.model_resnet import resnet_Model
from package.china_steel.china_steel_main import pred
from PIL import Image

app = Flask(__name__)

@app.route('/')
def index():
  return render_template('home.html')

@app.route('/home')
def home():
  return render_template('home.html')

@app.route('/yolo')
def yolo():
  return render_template('yolo.html')

@app.route('/chinasteel')
def chinasteel():
  return render_template('chinasteel.html')

@app.route('/yolo_index', methods=['GET', 'POST'])
def result_yolo():
  if request.method == 'POST':
      image = request.files["img"]
      if(image.filename==''):
        return render_template('yolo.html')
      predict_save(image)
      return render_template('yolo_result.html')

@app.route('/chinasteel_index', methods=['GET', 'POST'])
def result_chinasteel():
  if request.method == 'POST':
      image = request.files["img"]
      if(image.filename==''):
        return render_template('chinasteel.html')
      image = Image.open(image.stream)
      image.save('static/chinasteel_img.png')
      result = pred(image)
      return render_template('chinasteel_result.html', result = result)
if __name__ == '__main__':
  app.run()