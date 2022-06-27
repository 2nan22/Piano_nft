import cv2
from tqdm import tqdm
import glob
import hand_draw_extract as hde
import os
import numpy as np
from datetime import datetime

frame = hde.frame
width = hde.width
height = hde.height

# array를 하나씩 불러와서 튜플 형태로 입히는 작업
def array_to_tuple(f_no):

    a = frame[f_no]
    b = frame[f_no + 1]

    for i in range(len(a)):
        globals()[f'lf_line{i}'] = (int(a[i][0] * width), int(a[i][1] * height))

    for i in range(len(b)):
        globals()[f'rg_line{i}'] = (int(b[i][0] * width), int(b[i][1] * height))

# line drawing 및 save
def draw_sav_img(id):

    img = np.full((int(height), int(width), 3), (255, 255, 255), dtype=np.uint8)
    # img = cv2.imread('a_.jpg', cv2.IMREAD_UNCHANGED)


    cv2.line(img, lf_line0, lf_line1, (255, 150, 0))
    cv2.line(img, lf_line0, lf_line17, (255, 150, 0))  
    cv2.line(img, lf_line1, lf_line2, (255, 150, 0))
    cv2.line(img, lf_line2, lf_line3, (255, 150, 0))
    cv2.line(img, lf_line3, lf_line4, (255, 150, 0))
    cv2.line(img, lf_line5, lf_line6, (255, 150, 0))
    cv2.line(img, lf_line6, lf_line7, (255, 150, 0))
    cv2.line(img, lf_line7, lf_line8, (255, 150, 0))
    cv2.line(img, lf_line9, lf_line10, (255, 150, 0))
    cv2.line(img, lf_line10, lf_line11, (255, 150, 0))
    cv2.line(img, lf_line11, lf_line12, (255, 150, 0))
    cv2.line(img, lf_line13, lf_line14, (255, 150, 0))
    cv2.line(img, lf_line14, lf_line15, (255, 150, 0))
    cv2.line(img, lf_line15, lf_line16, (255, 150, 0))
    cv2.line(img, lf_line17, lf_line18, (255, 150, 0))
    cv2.line(img, lf_line18, lf_line19, (255, 150, 0))
    cv2.line(img, lf_line19, lf_line20, (255, 150, 0))


    cv2.line(img, rg_line0, rg_line1, (0, 0, 255))
    cv2.line(img, rg_line0, rg_line17, (0, 0, 255))
    cv2.line(img, rg_line1, rg_line2, (0, 0, 255))
    cv2.line(img, rg_line2, rg_line3, (0, 0, 255))
    cv2.line(img, rg_line3, rg_line4, (0, 0, 255))
    cv2.line(img, rg_line5, rg_line6, (0, 0, 255))
    cv2.line(img, rg_line6, rg_line7, (0, 0, 255))
    cv2.line(img, rg_line7, rg_line8, (0, 0, 255))
    cv2.line(img, rg_line9, rg_line10, (0, 0, 255))
    cv2.line(img, rg_line10, rg_line11, (0, 0, 255))
    cv2.line(img, rg_line11, rg_line12, (0, 0, 255))
    cv2.line(img, rg_line13, rg_line14, (0, 0, 255))
    cv2.line(img, rg_line14, rg_line15, (0, 0, 255))
    cv2.line(img, rg_line15, rg_line16, (0, 0, 255))
    cv2.line(img, rg_line17, rg_line18, (0, 0, 255))
    cv2.line(img, rg_line18, rg_line19, (0, 0, 255))
    cv2.line(img, rg_line19, rg_line20, (0, 0, 255))

    cv2.imwrite('./test_img/' + str(f'img{id}') +'.jpg', img)
    

def create_video():

    img_array = []
    now_time = datetime.now()
    current_time = now_time.strftime('%m%d_%H%M')
   
    for filename in tqdm(glob.glob(os.getcwd() + '/test_img/*.jpg')):
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    out = cv2.VideoWriter(current_time + '_result.avi', cv2.VideoWriter_fourcc(*'DIVX'), int(hde.video_fps), size)

    for i in tqdm(range(len(img_array))):
        out.write(img_array[i])
    out.release()
