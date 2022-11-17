import torch
import package.china_steel.model_p
from package.china_steel.model_p import MY_Model
import package.china_steel.data_file
import numpy as np
import package.china_steel.predict

"""
print("The first image's shape in dataset_train :", data_file.testing_data('data\china_steel\public_testing_data\__KUtRNYNzddFvQUe9DVhLTS23KlJk.jpg').size())
print("The first image's shape in dataset_train :", data_file.testing_data('data\china_steel\public_testing_data\__KUtRNYNzddFvQUe9DVhLTS23KlJk.jpg'))
images = data_file.testing_data('data\china_steel\public_testing_data\__KUtRNYNzddFvQUe9DVhLTS23KlJk.jpg')
array1=images.numpy()
mat=np.uint8(array1)
mat=mat.transpose(1,2,0)
print(mat.shape)
cv2.imshow('x',mat)
cv2.waitKey()
"""

import package.china_steel.model_resnet
from package.china_steel.model_resnet import resnet_Model
from torch.utils.data import DataLoader
def pred(path):
    image = np.array(path)
    images =package.china_steel.data_file.testing_data(image)
    test = []
    for i in range(100):
        test.append(images)
    test_loader = DataLoader(test, batch_size=100, shuffle=False, drop_last=False)
    result = package.china_steel.predict.predict(test_loader)
    return result[0]


"""
# data
dataset_train = data_file.public_data_getpoint('/content/Training Label/public_training_data.csv', mode='train')
dataset_val = data_file.public_data_getpoint('/content/Training Label/public_testing_data.csv', mode='val')
dataset_test = data_file.public_data_getpoint('/content/submission_template.csv', mode='test')
from torch.utils.data import DataLoader
train_loader = DataLoader(dataset_train, batch_size=100, shuffle=True, drop_last=True)
val_loader = DataLoader(dataset_val, batch_size=100, shuffle=False, drop_last=True)
test_loader = DataLoader(dataset_test, batch_size=100, shuffle=False, drop_last=False)

# train
max_epochs = 3
log_interval = 1
train_sco_list = []
train_loss_list = []
val_sco_list = []
val_loss_list = []

for epoch in range(1, max_epochs + 1):
  start_time = train.time.time()
  train_sco, train_loss = train(train_loader, model, train.criterion, train.optimizer)
  val_sco, val_loss = train.val(val_loader, model, train.criterion)
  end_time = train.time.time()
  epoch_mins, epoch_secs = train.epoch_time(start_time, end_time)
  
  train_sco_list.append(train_sco)
  train_loss_list.append(train_loss)
  val_sco_list.append(val_sco)
  val_loss_list.append(val_loss)

  if epoch % log_interval == 0:
    print('=' * 20, 'Epoch', epoch, '=' * 20)
    print(f'Epoch Time: {epoch_mins}m {epoch_secs}s')
    print('Train Sco: {:.6f} Train Loss: {:.6f}'.format(train_sco, train_loss))
    print('  Val Sco: {:.6f}   Val Loss: {:.6f}'.format(val_sco, val_loss))
  torch.save(model, "picachu_model_gru.pth")

# plot training info
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))
plt.plot(range(len(train_sco_list)), train_sco_list)
plt.plot(range(len(val_sco_list)), val_sco_list, c='r')
plt.legend(['train', 'val'])
plt.title('Score')
plt.show()

plt.figure(figsize=(12, 4))
plt.plot(range(len(train_loss_list)), train_loss_list)
plt.plot(range(len(val_loss_list)), val_loss_list, c='r')
plt.legend(['train', 'val'])
plt.title('Loss')
plt.show()

# output csv
output_csv = predict(test_loader, model)
picname_csv = []
with open("/content/submission_template.csv", newline='') as csvfile:
  reader = csv.DictReader(csvfile)
  for row in reader:
    picname_csv.append(row['id'])
with open('result.csv', 'w', newline='') as csvFile:
  writer = csv.DictWriter(csvFile, fieldnames=['id', 'text'])
  writer.writeheader()
  for i in range(len(output_csv)):
    writer.writerow({'id':picname_csv[i], 'text':output_csv[i]})
"""