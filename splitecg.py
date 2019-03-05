import numpy as np
import wfdb
import os

import shutil

AAMI_label = [{'N','.','L','R','e','j'},{'A','a','J','S'},{'V','E'},{'F'},{'f','/','Q'}]
AAMI_label_name = ['N','S','V','F','Q']
file_list = []
data_dir = 'oridata/'

handle_root = 'signaldata1/'
list_train_path = 'index_dataset/trainlist.txt'
list_test_path = 'index_dataset/testlist.txt'
fp1 = open(list_train_path ,'w')
fp2 = open(list_test_path ,'w')


def get_aami_label_number(s1):
    for idconf , i in enumerate(AAMI_label):
        if(s1 in i):
            return idconf
    return -1

def convert_singal(oral_single , downsampled = 64):
    len_t = len(oral_single)
    oral_r = np.linspace(1,len_t,len_t)
    now_r = np.linspace(1,len_t , downsampled)
    return np.interp(now_r , oral_r  , oral_single)

def  writerecord(save_root,savefilename , writearray):

    ftp = open(os.path.join(save_root,savefilename),'w')
    for line in writearray:
        ftp.write('{:.4f} '.format(line))
    ftp.close()

for fname in os.listdir(data_dir):
    words = fname.split('.')
    if(words[0] in file_list):
        continue
    else:
        print(words[0])
        file_list.append(words[0])
        tfilename = os.path.join(data_dir , words[0])
        s1 = wfdb.rdsamp(tfilename)
        ann = wfdb.rdann(tfilename , 'atr')
        II = np.array(s1[0][:,0])
        label_pos = ann.sample
        label_name = ann.symbol
        for idk , ikey in enumerate(label_pos):
            tlabel = get_aami_label_number(label_name[idk])
            if(tlabel == -1):
                continue
            if(idk < len(label_pos) -5):
                onset = (ikey + label_pos[idk+1]) //2
                offset = (label_pos[idk+1]+ label_pos[idk+2])//2
                new_heartbeat = convert_singal(II[onset: offset])
                onefilename = '{}_{}.txt'.format(words[0],idk)
                randomx = np.random.randint(0,10)
                cpath = os.path.join(handle_root , '{}_{}.txt'.format(words[0],idk))
                if randomx <= 7:
                    fp1.write('{} {}\n'.format(cpath,tlabel))
                else:
                    fp2.write('{} {}\n'.format(cpath, tlabel))

                writerecord(handle_root,onefilename,new_heartbeat)
            #break
        # break

fp1.close()
fp2.close()

