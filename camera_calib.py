import json
import os
import matplotlib.pyplot as plt

cdir = os.getcwd()

fdir = os.path.join(cdir, 'data', 'RJDataset', 'labels', '3D_mouth_detections', 'camera_calibration_data')

fls = os.listdir(fdir)

cx = []
cy = []
cam = []


for filename in fls:

    idx = filename.find('.json')
    cam.append(int(filename[idx-2:idx]))

    f_name = os.path.join(fdir, filename)
    f = open(f_name)
    data = json.load(f)

    cx.append(round(float(data['camera']['cx']), 2))
    cy.append(round(float(data['camera']['cy']), 2))

cx = cx[:11]
cy = cy[:11]
cam = cam[:11]

fig, ax = plt.subplots()
ax.scatter(cx, cy)

for i, txt in enumerate(cam):
    ax.annotate(txt, (cx[i], cy[i]))

plt.show()




    