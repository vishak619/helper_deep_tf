def TBCallback(dir_name,expname):
  """
  This function is used to retuen a call back fuction object pertaining to
  tensorboard callback
  Args
  dir_name = The directory name for saving the log files
  exp_nmae = the name of the experiment
  Returns:
  A tensor board call back object that logs the file in to the directory
  dir_name/expname/timestamp
  where timestamp is timestamp of exp's date

  
  """
  import datetime
  log_dir = dir_name +'/' + expname + '/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  import tensorflow as tf
  tbcallb = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
  print(f"Saving tensor board log files to {log_dir}")
  return tbcallb
def download_dataset_unzip(url):
  """
  Download the dataset(zip) from the url and unzips it 


  """
  import requests
  URL = url
  response = requests.get(URL)
  help_list = url.split("/")
  file_name = help_list[-1]
  # 3. Open the response into a new file called instagram.ico
  open(file_name, "wb").write(response.content)
  print(f'Downloaded {file_name} from {url}') 

  import zipfile
  zip_ref = zipfile.ZipFile(file_name)
  zip_ref.extractall()
  print(f'Unzipped {file_name}') 
  zip_ref.close()
def plot_results(history_1):

  """
  plot the validastion loss versus epochs and accuracy versus epochs in
  two plots


  """
  import pandas as pd
  import matplotlib.pyplot as plt
  histor_1_pd = pd.DataFrame(history_1.history)
  fig_1 = plt.figure('figure_1')
  epochs = range(len(history_1.history['loss']))
  plt.plot(epochs, history_1.history['loss'], color ='red')
  plt.plot(epochs, history_1.history['val_loss'], color = 'green')
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.legend(['loss', 'Val_loss'])
  plt.show()
  fig_2 = plt.figure('figure_2')
  plt.plot(epochs, history_1.history['accuracy'], color = 'red')
  plt.plot(epochs, history_1.history['val_accuracy'], color = 'green')
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy')
  plt.legend(['Accuracy', 'Val_Accuracy'])
  plt.show()
def modelcheckpoint(dir_name, exp_name):

  """
  This function is used to retuen a call back fuction object pertaining to
  Model Checkpoint callback
  Args
  dir_name = The directory name for saving the log files
  exp_nmae = the name of the experiment
  Returns:
  A modelcheck point call back object that logs the file in to the directory
  dir_name/expname/timestamp
  where timestamp is timestamp of exp's date
  save every epoch with best model turned to false
  save weights only =True


  """
  import tensorflow as tf
  import datetime
  path = dir_name + "/" + exp_name + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/" + "checkpoint.ckpt"
  check_callback = tf.keras.callbacks.ModelCheckpoint(path, monitor ='val_loss', save_weights_only = True,
                                                      save_freq='epoch', verbose = 1)
  return check_callback
def compare_historys(original_history, new_history, initial_epochs=5):
    """
    Compares two model history objects. only in case of transfer learning trained initially 
    and then fine tuning
    """
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    # Get original history measurements
    acc = original_history.history["accuracy"]
    loss = original_history.history["loss"]

    print(len(acc))

    val_acc = original_history.history["val_accuracy"]
    val_loss = original_history.history["val_loss"]

    # Combine original history with new history
    total_acc = acc + new_history.history["accuracy"]
    total_loss = loss + new_history.history["loss"]

    total_val_acc = val_acc + new_history.history["val_accuracy"]
    total_val_loss = val_loss + new_history.history["val_loss"]

    print(len(total_acc))
    print(total_acc)

    # Make plots
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(total_acc, label='Training Accuracy')
    plt.plot(total_val_acc, label='Validation Accuracy')
    plt.plot([initial_epochs-1, initial_epochs-1],
              plt.ylim(), label='Start Fine Tuning') # reshift plot around epochs
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(total_loss, label='Training Loss')
    plt.plot(total_val_loss, label='Validation Loss')
    plt.plot([initial_epochs-1, initial_epochs-1],
              plt.ylim(), label='Start Fine Tuning') # reshift plot around epochs
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()
  
def visualise_random_img(dirl,classl):
  """
  Visualise Visualise Visualise
  This function can be used to visualise a random image in training datset
  to become one with the dataset
  Args:
  dirl = the trainig directory path of dataset
  classl =  the classname for visualising the random image
  if you want to visualise a random image from any class
  just use "any" as classl
  



  """
  import matplotlib.pyplot as plt
  import pathlib
  import numpy as np
  import matplotlib.image as mpimg
  import random
  import os
  if classl=="any":
    data_dir = pathlib.Path(dirl)
    class_names = np.array(sorted([item.name for item in data_dir.glob("*")]))
    rs = random.randint(0,len(class_names)-1)
    classl = class_names[rs]
  folder = dirl + "/" + classl
  rimg= random.sample(os.listdir(folder),1)
  img = mpimg.imread(folder +"/" + rimg[0])
  plt.imshow(img)
  plt.title(classl)
  plt.axis("off")
  print(f"Shape : {img.shape}")




