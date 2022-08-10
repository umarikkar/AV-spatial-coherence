import pandas as pd
import csv
import os

import numpy as np

def csv_to_list(csv_path):
    as_list = None
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        as_list = list(reader)
    return as_list



def get_metadata(current_path, test_file, speech_file):

        """
        the array columns will be in the form:

        0:seq_name | 1:time | 2:x | 3:pseudo_label | 4:SA_total | 5:SA_male | 6:SA_female | 7:x_male | 8:y_male | 9:x_female | 10:y_female

        'x' is the pseudo label confidence, SA is speaker activity
        
        """

        def set_speak(arr, pairs, current_idxs, col):
                for pair in pairs:
                        pair = pair + current_idxs[0]
                        p1 = np.ceil(pair[0]).astype(int)
                        p2 = np.ceil(pair[1]).astype(int)
                        arr[p1:p2, col] = 'SPEAKING'
                
                return arr



        columns = ['name', 'time', 'x', 'pseudo-label', 'speech activity', 'SA_male', 'SA_female', 'x_male', 'y_male', 'x_female', 'y_female']

        sub_folder = os.path.join(current_path, 'data', 'RJDataset', 'labels', '3D_mouth_detections')

        csv_original = csv_to_list(test_file)[1:]

        csv_array = np.array(csv_original)
        names = csv_array[:,0]

        # let us first append the list with the two extra columns
        sz = csv_array.shape[0]
        empty_arr = np.empty((sz,6), dtype=None) # male_bool, male_x, male_y, female_bool, female_x, female_y
        csv_array = np.append(csv_array, empty_arr, axis=-1)
        csv_array[:, 5:7] = 'NOT_SPEAKING'

        
        # ADDING THE COORDINATES!
        test_seqs = ['conversation1_t3', 'femalemonologue2_t3', 'interactive1_t2', 'interactive4_t3', 'malemonologue2_t3']

        male = [True, False, True, True, True]
        female = [True, True, True, True, False]

        cams = ['01', '02', '03', '04', '05', 
                '06', '07', '08', '09', '10',
                '11', '12', '13', '14', '15',
                '16', '17', '18', '19', '20',
                '21', '22']

        # now let us load the xls file with the speech sequences.
        xls = pd.ExcelFile(speech_file)

        for seq_idx, seq in enumerate(test_seqs):

                df1 = pd.read_excel(xls, seq)

                R_frames = np.array([df1['Unnamed: 2'].array[1:], df1['Unnamed: 3'].array[1:]]).transpose()
                J_frames = np.array([df1['Unnamed: 6'].array[1:], df1['Unnamed: 7'].array[1:]]).transpose()

                R_frames = R_frames[~np.all(R_frames == 0, axis=1)]
                J_frames = J_frames[~np.all(J_frames == 0, axis=1)]

                for cam in cams:

                        seq_cam = seq+'-cam'+cam
                        sub_file = os.path.join(sub_folder, seq, '2D_projections', seq_cam +'_GT.csv')
                        csv_sub = csv_to_list(sub_file)
                        sub_array = np.array(csv_sub)

                        coord = sub_array[:, 1:3].astype('float')

                        current_idxs = np.sort(np.where(names==seq_cam))[0]
                        
                        if current_idxs.shape[0] != 0:

                                # add the speaker activity indexes.
                                csv_array = set_speak(csv_array, R_frames, current_idxs, 5)
                                csv_array = set_speak(csv_array, J_frames, current_idxs, 6)

                                # for pair in R_frames:
                                #         pair = pair + current_idxs[0]
                                #         p1 = np.ceil(pair[0]).astype(int)
                                #         p2 = np.floor(pair[1]).astype(int)
                                #         csv_array[p1:p2, 5] = 'SPEAKING'

                                # for pair in J_frames:
                                #         pair = pair + current_idxs[0]
                                #         csv_array[np.ceil(pair[0]):np.floor(pair[1]), 6] = 'SPEAKING'


                                # add the coordinates.
                                if male[seq_idx] and female[seq_idx]:
                                        m_idxs = np.sort(np.where(sub_array[:,-1]==str(1)))[0]
                                        f_idxs = np.sort(np.where(sub_array[:,-1]==str(2)))[0]

                                        csv_array[current_idxs, -4:-2] = coord[m_idxs]
                                        csv_array[current_idxs, -2:] = coord[f_idxs]

                                elif male[seq_idx]:
                                        csv_array[current_idxs, -4:-2] = coord
                                else:
                                        csv_array[current_idxs, -2:] = coord


        return csv_array, columns




if __name__ == "__main__":

        current_path = os.getcwd()
        test_path = os.path.join(current_path, 'data', 'csv')
        test_file = os.path.join(test_path, 'test.csv')
        save_file = os.path.join(test_path, 'test_new.csv')
        speech_file = os.path.join(current_path, 'data', 'RJDataset', 'labels', 
                                        'speech_activity','speech_activity.xlsx')

        csv_array, columns = get_metadata(current_path, test_file, speech_file)

        csv_df = pd.DataFrame(csv_array, columns=columns)

        csv_df.to_csv(save_file, index=False)


        pass

        