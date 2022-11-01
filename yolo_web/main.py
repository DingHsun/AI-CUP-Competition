import numpy as np
import cv2
import torch
import os

def retxt(path, height, width):
    if os.stat(path).st_size == 0:
        return
    data = open(path)
    list_of_lists = []
    for line in data:
        stripped_line = line.strip()
        ob_list = [np.float(s) for s in stripped_line.split(',')]
        ob_list[0] = np.int32(ob_list[0])
        ob_list[1] = (ob_list[1]+(ob_list[3]/2))/width
        ob_list[2] = (ob_list[2]+(ob_list[4]/2))/height
        ob_list[3] = ob_list[3]/width
        ob_list[4] = ob_list[4]/height
        list_of_lists.append(ob_list)
    np.savetxt(path, list_of_lists, fmt='%d %f %f %f %f',
               delimiter=' ', newline='\n')

'''
for i in range(1, 801, 1):
	if (i < 10):
		img = "img000" + str(i) + ".png"
		text = "img000" + str(i) + ".txt"
	elif (i < 100):
		img = "img00" + str(i) + ".png"
		text = "img00" + str(i) + ".txt"
	else:
		img = "img0" + str(i) + ".png"
		text = "img0" + str(i) + ".txt"
	height, width, _ = cv2.imread('./content/train/' + str(img)).shape
	retxt('./content/train/' + str(text), height, width)
	os.replace('./content/train/' + str(img),
			'./content/train/images/' + str(img))
	os.replace('./content/train/' + str(text),
			'./content/train/labels/' + str(text))
for i in range(801, 1001, 1):
	if (i < 1000):
		img = "img0" + str(i) + ".png"
		text = "img0" + str(i) + ".txt"
	else:
		img = "img" + str(i) + ".png"
		text = "img" + str(i) + ".txt"
	height, width, _ = cv2.imread('./content/train/' + str(img)).shape
	retxt('./content/train/' + str(text), height, width)
	os.replace('./content/train/' + str(img), './content/val/images/' + str(img))
	os.replace('./content/train/' + str(text),
			'./content/val/labels/' + str(text))
'''

model = torch.hub.load('ultralytics/yolov5', 'custom' , path='best.pt')

result = model('./content/public/img1022.png')

result.show()