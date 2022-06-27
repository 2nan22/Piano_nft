import os
from tqdm import tqdm
import convert2video as cvt
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
    

        # 최소 감지 신뢰도(0~1) / 최소 추적 신뢰도(0~1)
min_detect = input('최소 감지 신뢰도(0 ~ 1) 숫자를 입력하세요 ' + '\n' + '(default = 0.5)' + ' : ')
min_tracking = input('최소 추적 신뢰도(0 ~ 1) 숫자를 입력하세요' + '\n' + '(default = 0.5)' +  ' : ')

hde.hand_draw_extract(float(min_detect), float(min_tracking))

array_lst = hde.frame

for idx in tqdm(range(len(array_lst)-1)):
    cvt.array_to_tuple(idx)
    cvt.draw_sav_img(idx)

cvt.create_video()