import os

import cv2
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

dataset = 'RJDataset'

base_path = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(os.path.dirname(base_path), dataset)

data_type = 'videos'
take = 'conversation1_t1'
vid_index = 0

time=5
fps=30

load_vid=False

def main():

    frames = []
    if load_vid:

        vidFolder = os.path.join(os.path.join(dataset_path, data_type, take))
        all_files = os.listdir(vidFolder)

        vid_files = [file for file in all_files if file.endswith('.mp4')]

        vFile = vid_files[vid_index]
        # print('v file hahaha', vFile)

        vFile_path = os.path.join(vidFolder, vFile)

        path = vFile_path
        cap = cv2.VideoCapture(path)
        ret = True
        while ret:
            ret, img = cap.read() # read one frame from the 'capture' object; img is (H, W, C)
            if ret:
                frames.append(img)
            if len(frames) >= time*fps:
                ret=False

        video = np.stack(frames, axis=0)

        for idx, frame in enumerate(frames):
            frame = cv2.resize(frame, (256, 256))
            cv2.imwrite(os.path.join(base_path, 'frames', 'test_frame_'+str(idx)+'.png'), frame) 
    
    else:
        for idx in range(time*fps):
            frames.append(cv2.imread(os.path.join(base_path, 'frames', 'test_frame_'+str(idx)+'.png')))

        video = np.stack(frames, axis=0)


    vid_normalised = np.zeros_like(video)
    vid_mean = np.zeros_like(video[0].squeeze())
    for ch in range(3):
        vid = video[:,:,:,ch].squeeze()
        mn = np.mean(vid, axis=0)
        vid_mean[:,:,ch] = mn
        vid_mn = vid - mn
        print(vid.max())
        vid_mn = 255*(vid_mn - np.min(vid_mn)) / np.max(vid_mn)
        vid_normalised[:,:,:,ch] = vid_mn

    print(vid_normalised.shape)

    f_num = 50

    img = video[f_num]
    img_norm = vid_normalised[f_num]

    img_mat = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_mean = cv2.cvtColor(vid_mean, cv2.COLOR_BGR2RGB)
    img_norm_mat = cv2.cvtColor(img_norm, cv2.COLOR_BGR2RGB)

    plt.figure()
    plt.subplot(131)
    plt.imshow(img_mat), plt.axis('off'), plt.title('(a)')
    plt.subplot(132)
    plt.imshow(img_mean), plt.axis('off'),plt.title('(b)')
    plt.subplot(133)
    plt.imshow(img_norm_mat), plt.axis('off'),plt.title('(c)')
    plt.show()

    print(img_mat.shape, img_norm_mat.shape)


    


    # img = cv2.imread('test_img.png')

    # img_mean = np.mean(img, axis=-1)
    # print(img_mean)

    pass



if __name__ == "__main__":
    main()