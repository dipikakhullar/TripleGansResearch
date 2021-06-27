import numpy as np 
import pandas as pd
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2 
TRAIN_TEST_SPLIT = 0.7
IM_WIDTH = IM_HEIGHT = 198


def parse_dataset(dataset_path, ext='jpg'):
    """
    Used to extract information about our dataset. It does iterate over all images and return a DataFrame with
    the data (age, gender and sex) of all files.
   aaaa """
    def parse_info_from_file(path):
        """
        Parse information from a single file
        """
        try:
            filename = os.path.split(path)[1]
            filename = os.path.splitext(filename)[0]
            age, gender, race, _ = filename.split('_')

            return int(age), dataset_dict['gender_id'][int(gender)], dataset_dict['race_id'][int(race)]
        except Exception as ex:
            return None, None, None
        
    files = glob.glob(os.path.join(dataset_path, "*.%s" % ext))
    
    records = []
    for file in files:
        info = parse_info_from_file(file)
        records.append(info)
        
    df = pd.DataFrame(records)
    df['file'] = files
    df.columns = ['age', 'gender', 'race', 'file']
    df = df.dropna()
    
    return df


def image_to_tensor(file_path_list, size = (3, 224, 224)):
    output = []
    for file_path in file_path_list:
        im= np.asarray(Image.open(file_path).convert('RGB'))
#         print(im.shape)
        im=cv2.resize(im, dsize=(size[1],size[2]))
        # im = np.reshape(im, size)
        output.append(im)
    return output


def load_utk_data(data_dir):
    data_dir = 'UTKFace'
    print(os.listdir(data_dir))
    df = pd.read_csv('utk_face.csv')
    print(len(df))
    x_data = image_to_tensor(df['file'])
    y_data = df['age']
    x_train = x_data[:20000]
    y_train = y_data[:20000]

    x_test = x_data[20000:]
    y_test = y_data[20000:]
    print(np.shape(x_train))
    return x_train, y_train, x_test, y_test




