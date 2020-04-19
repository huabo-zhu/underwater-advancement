import os
import numpy as np
import cv2
import natsort
import datetime
from RefinedTramsmission import Refinedtransmission
from getAtomsphericLight import getAtomsphericLight
from getGbDarkChannel import getDarkChannel
from getTM import getTransmission
from sceneRadiance import sceneRadianceRGB

np.seterr(over='ignore')
if __name__ == '__main__':
    pass

# folder = "C:/Users/Administrator/Desktop/UnderwaterImageEnhancement/Physical/UDCP"
# folder = "C:/Users/Administrator/Desktop/Databases/Dataset"
folder = "D:\\img"
path = folder + "\\InputImages"
files = os.listdir(path)
for file in files:
    print(file)
files = natsort.natsorted(files)
# 开始计时
starttime = datetime.datetime.now()
for i in range(len(files)):
    file = files[i]
    filepath = path + "\\" + file
    prefix = file.split('.')[0]
    if os.path.isfile(filepath):
        print('********    file   ********', file)
        img = cv2.imread(folder +'\\InputImages\\' + file)

        # cv2.imshow('image', img)
        print('img',img)

        blockSize = 9
        GB_Darkchannel = getDarkChannel(img, blockSize)
        AtomsphericLight = getAtomsphericLight(GB_Darkchannel, img)
        print('AtomsphericLight', AtomsphericLight)
        # print('img/AtomsphericLight', img/AtomsphericLight)

        # AtomsphericLight = [231, 171, 60]

        transmission = getTransmission(img, AtomsphericLight, blockSize)

        # cv2.imwrite('OutputImages/' + prefix + '_UDCP_Map.jpg', np.uint8(transmission))

        transmission = Refinedtransmission(transmission, img)
        sceneRadiance = sceneRadianceRGB(img, transmission, AtomsphericLight)
        """
        cv2.imshow('image', sceneRadiance)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        """
        # print('AtomsphericLight',AtomsphericLight)
        # cv2.imwrite("D:\\img", np.uint8(transmission* 255))
        # cv2.imwrite(folder + '\\OutputImages\\' + prefix+'UDCP_TM.jpg', np.uint8(transmission* 255))
        cv2.imwrite(folder + '\\OutputImages\\' + prefix + '_UDCP.jpg', sceneRadiance)
Endtime = datetime.datetime.now()
print('Time:', Endtime - starttime)


