#  绘制多个模型对于同一问题的PR曲线

import matplotlib.pyplot as plt
import numpy as np
import os
num = 0

root_path = '/train/results/ecg_cls/'
# compare_path = ['38attribute/resnet50_22w_16k','38attribute/resnet50_38w_16K','38attribute/resnet50_39.4w_16K','38attribute/resnet50_49.4w_16k']
# compare_path = ['38attribute/resnet50_39.4w_16K','37attribute/resnet50_50w_16k','38attribute/resnet50_49.4w_16k']
#compare_path = ['resnet50_22w_16k','resnet50_35w_16k','resnet50_45w_16k','resnet50_50w_16k']
# compare_path = ['38attribute/resnet50_39.4w_16K','37attribute/resnet50_22w_16k']
#compare_path = ['mixupresnet18_12w','resnet_18_40w','resnet50_50w']
compare_path = ['results/',]
save_path = 'PR/'


if(os.path.exists(save_path)):
    for path in os.listdir(save_path):
        os.remove(os.path.join(save_path,path))
        print('{} has been deleted'.format(path))
else:
    os.mkdir(save_path)
    print('{} has been created'.format(save_path))
name_person_attr = ['双肩背包','单肩包','行李箱','眼镜','帽子','口罩','上身_白',
                   '上身_灰','上身_黑','上身_绿','上身_蓝','上身_红','上身_紫',
                    '上身_黄','上身_粉','上身_橘','上身_棕','上身_彩','下身_白',
                    '下身_灰','下身_黑','下身_绿','下身_蓝','下身_红','下身_紫',
                    '下身_黄','下身_粉','下身_橘','下身_棕','下身_彩','年龄',
                    '上身_格子','上身_条纹','下身_短裤','下身_短裙','下身_长裤',
                    '性别','短袖']
name_person_attr = ['phone','smoke']
name_person_attr = ['cad',]
count_classes = 1
num_classes = [5,]

threshold = [0.01, 0.03, 0.05, 0.08, 0.1, 0.15, 0.20, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.825, 0.85, 0.9, 0.922, 0.94, 0.96, 0.980, 0.985, 0.990, 0.995, 0.999]
color_line = ['orange' ,'green' ,'red' , 'blue']
color_point = ['blue' , 'red' , 'green' , 'yellow']
color_th = ['red' ,'blue' , 'green','black']

tulix = np.array([0.1,0.2],dtype = float)
tuliheight = np.array([0.04,0.04],dtype = float)


for no_pa in range(count_classes):
    for i_cls in range(num_classes[no_pa]):
    
        tsavepath = os.path.join(save_path, '{}_{}.png'.format(no_pa , i_cls))
        print(tsavepath)
        plt.figure(figsize=(8, 8))
        plt.grid(True)
        plt.axis([0, 1, 0, 1])
        plt.ylabel('Precision  TP/(TP+FP)')
        plt.xlabel('Recall  TP/(TP+FN)')
        plt.xticks(np.arange(0, 1, 0.1))
        plt.yticks(np.arange(0, 1, 0.1))
        plt.title('{}_{}'.format(name_person_attr[no_pa],i_cls))
        for idk,file_list in enumerate(compare_path):
            filename = os.path.join(root_path ,file_list,'result_{}.txt'.format(no_pa))
            if(os.path.exists(filename) == False):
                continue
            labels = []
            scores = []
            fp = open(filename ,'r', encoding='UTF-8')
            for line in fp:
                l = line.strip().split(' ')
                tempu = int(l[1])
                if( int(l[1]) == -1):
                    continue
                else:
                    if(int(l[1]) == i_cls):
                        labels.append(1)
                    else:
                        labels.append(0)
                    scores.append(float(l[3+i_cls]))
                # labels.append(int(l[1]))
            labels = np.array(labels)
            scores = np.array(scores)
            fp.close()

            precisions = []
            recalls = []
            mine_0_5 = []
            myprecision ,myrecall,myth  = 0 , 0 , 0
            for th in threshold:
                predict = scores > th
                trueset = predict == labels
                falseset = 1 - trueset
                TPset = trueset & predict
                FPset = predict & falseset
                TNset = trueset & (1 - predict)
                FNset = falseset & (1 - predict)
                TP = np.sum(TPset, axis=0).astype(np.float32)
                FP = np.sum(FPset, axis=0).astype(np.float32)
                TN = np.sum(TNset, axis=0).astype(np.float32)
                FN = np.sum(FNset, axis=0).astype(np.float32)
                TP_rate = 0 if TP == 0 else TP / (TP + FN)
                Recall = TP_rate
                Precision = 1 if TP == 0 else TP / (TP + FP)

                if th == 0.5:
                    mine_0_5.append(Recall)
                    mine_0_5.append(Precision)
                precisions.append(Precision)
                recalls.append(Recall)

           # print(name_person_attr[no_pa])
            plt.plot(recalls, precisions, label='Default', color=color_line[idk])
            plt.plot(recalls, precisions, 'o', color=color_line[idk])
            # plt.plot([mine_0_5[0]], [mine_0_5[1]], '*', color=color_point[idk], markersize=12)
            new_height = tuliheight * (1 + idk)
            plt.plot(tulix , new_height , color = color_line[idk])

            # plt.text(tulix[1],new_height[0],compare_path[idk].strip().split('/')[-1])
            plt.text(tulix[1], new_height[0], compare_path[idk])
            for idxx, x, y in zip(range(len(precisions)), recalls, precisions):
                # if idxx == 12: print('thres =', threshold[idxx], 'R, P =', x, y)
                # if idxx == 22: print('thres =', threshold[idxx], 'R, P =', x, y)
                if idxx % 2 == 0:
                    plt.text(x + 0.02, y + 0.02, '%d(%.3f)' % (idxx, threshold[idxx]), rotation=60, ha='center', va='bottom',
                             fontsize=7)
        #     plt.title(true_labels[idk])


        # plt.plot(tulix  , new_height, labels='Default', color='b')
        plt.savefig(tsavepath)
        plt.close('all')

        # plt.show()
