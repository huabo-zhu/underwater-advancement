
import os
import numpy as np
import cv2
import natsort
import xlwt
from skimage import exposure
import datetime
from sceneRadianceCLAHE import RecoverCLAHE
from sceneRadianceHE import RecoverHE

np.seterr(over='ignore')
if __name__ == '__main__':
    pass
folder = "D:\\img"
path = folder + "\\InputImages"
files = os.listdir(path)
files =  natsort.natsorted(files)
# 开始计时
starttime = datetime.datetime.now()
for i in range(len(files)):
    file = files[i]
    filepath = path + "\\" + file
    prefix = file.split('.')[0]
    if os.path.isfile(filepath):
        print('********    file   ********',file)
        # img = cv2.imread('InputImages/' + file)
        img = cv2.imread(folder + '\\InputImages\\' + file)
        sceneRadiance = RecoverHE(img)
        cv2.imwrite(folder + '\\OutputImages\\' + prefix + '_HE.jpg', sceneRadiance)
Endtime = datetime.datetime.now()
print('Time:', Endtime - starttime)
