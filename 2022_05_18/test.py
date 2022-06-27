import os
import numpy as np
from tqdm import tqdm
import cvt
import hand_draw_extract as hde

###########################################################################
#### 새로운 Test용 video file 저장 시 test_OOO.mp4 형태로 Naming 바람########
###########################################################################

file_lst = './test_img'

if not os.path.exists(file_lst):
    os.mkdir(file_lst)

# img 폴더 reset
def resetfolder(filelst):
    
    print('-------------------------------------------------')
    print('img 폴더를 reset하는 중입니다.. 잠시만 기다려주세요..')
    print('-------------------------------------------------')
    
    if os.path.exists(filelst):
        for file in os.scandir(filelst):
            os.remove(file.path)
        return 'Remove All file'
    else:
        return 'Directory Not Found'


resetfolder(file_lst)
    

mc = hde.model_complex
md = hde.min_detect
mt = hde.min_tracking


hde.hand_draw_extract(int(mc), float(md), float(mt))


frame_l = hde.frame_l
frame_r = hde.frame_r

print('----------------------------------------------------------------------------------')
print('왼 손 길이 : ', len(frame_l), '\n','오른손 길이 : ', len(frame_r))
print('array to tuple & drawing image.....')

for idx in tqdm(range(max(len(frame_l), len(frame_r)))):
    if idx < len(frame_l) and idx < len(frame_r):
        cvt.array_to_tuple_l(frame_l, idx)
        cvt.array_to_tuple_r(frame_r, idx)
        cvt.draw_sav_img(idx)
    elif idx >= len(frame_l) and idx < len(frame_r):
        cvt.array_to_tuple_r(frame_r, idx)
        cvt.draw_sav_img(idx)
    elif idx >= len(frame_r) and idx < len(frame_l):
        cvt.array_to_tuple_l(frame_l, idx)
        cvt.draw_sav_img(idx)
    else:
        break

cvt.create_video()