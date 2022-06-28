import pandas as pd
from ast import literal_eval
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from PIL import Image, ImageDraw
import cv2
import glob

csv_path = './point_csv/0623_1254_sample1_full.csv'

def csv2array(csv_path=csv_path):

    test = pd.read_csv(csv_path)

    col = test.columns.tolist()

    # 문자열로 되어있는 값들을 벗겨내기
    for c in col:
        test[c] = test[c].apply(lambda x : literal_eval(x))

    num=0
    arr_name = []

    print('----------csv to array -------------')
    # 프레임 별 array 생성 
    for i in tqdm(range(len(test))):
        # 전역변수로 변수명을 'arr_ + 프레임 no' 선언 
        globals()[f'arr_{num}'] = []
        arr_name.append(f'arr_{num}')

        for c in col:
            if test.loc[i][c] != 0:
                # x, y 값을 array에 담고, 리스트에 append
                new_arr = np.array([test.loc[i][c]['x'], test.loc[i][c]['y']])
                globals()[f'arr_{num}'].append(new_arr)
            else:
                new_arr = np.array([test.loc[i][c], test.loc[i][c]])
                globals()[f'arr_{num}'].append(new_arr)
        # 21개의 좌표가 담긴 각각의 array를 하나의 array로 => Frame 별 array 생성
        globals()[f'arr_{num}'] = np.array(globals()[f'arr_{num}'])

        num += 1
    
    return arr_name

def array2img():

    arr_name = csv2array()

    draw_path = './pillow_img/'

    # img 저장 경로 파일 존재 시, 전부 삭제 후 코드 진행
    if os.path.exists(draw_path):
        for file in os.scandir(draw_path):
            os.remove(file.path)
    else:
        os.mkdir(draw_path)

    # 높이 / 너비 설정
    height = 1000
    width = 1000

    # 손꾸락 색깔 (살색)
    rgb = (255, 159, 127)

    # 초기값 설정
    mode = 0
    ent_mode = 0
    cnt = 0

    # 색상 초기값 설정
    color1 = 255
    color2 = 255
    color3 = 255

    # img crop
    area = (200, 300, 700, 800)

    print('----------array to image-------------')


    # for i in tqdm(range(len(arr_name))):
    for i in tqdm(range(100)):

        image = Image.new(mode='RGB', size=(height, width), color=(255, 255, 255))
        draw = ImageDraw.Draw(image, 'RGB')

        a = 10

        if ent_mode == 0:

            color2 = 255
            color3 = 255

            while a != 2000:
                x = a - 10
                y = 0
                for _ in range(a):
                    draw.rectangle([(x, y), (x+10, y+10)], fill=(color1, color2, color3), width=0)
                    x -= 10
                    y += 10
                a += 10

                # RGB 값 끝에 다다를 때 mode 변환
                if color1 == 155 :
                    mode = 1
                elif color1 == 255:
                    mode = 0

                # mode 별로 RGB 값을 + 할지 - 할지 고려
                if mode == 0:
                    color1 -= 1
                elif mode == 1:
                    color1 += 1

        elif ent_mode == 1:
            
            color1 = 255
            color3 = 255

            # 프레임 1EA 배경 생성
            while a != 2000:
                x = a - 10
                y = 0
                for _ in range(a):
                    draw.rectangle([(x, y), (x+10, y+10)], fill=(color1, color2, color3), width=0)
                    x -= 10
                    y += 10
                a += 10

                # RGB 값 끝에 다다를 때 mode 변환
                if color2 == 155 :
                    mode = 1
                elif color2 == 255:
                    mode = 0

                # mode 별로 RGB 값을 + 할지 - 할지 고려
                if mode == 0:
                    color2 -= 1
                elif mode == 1:
                    color2 += 1

        elif ent_mode == 2:

            color1 = 255
            color2 = 255

            while a != 2000:
                x = a - 10
                y = 0
                for _ in range(a):
                    draw.rectangle([(x, y), (x+10, y+10)], fill=(color1, color2, color3), width=0)
                    x -= 10
                    y += 10
                a += 10

                # RGB 값 끝에 다다를 때 mode 변환
                if color3 == 155 :
                    mode = 1
                elif color3 == 255:
                    mode = 0

                # mode 별로 RGB 값을 + 할지 - 할지 고려
                if mode == 0:
                    color3 -= 1
                elif mode == 1:
                    color3 += 1

        if cnt < 100:
            cnt += 1
        else:
            ent_mode += 1
            cnt = 0

        if ent_mode > 2:
            ent_mode = 0
            
        # 좌표 값 그리기
        for idx in range(42):

            p = np.trunc(np.multiply(eval(arr_name[i])[idx], np.array([1000, 1000])))
                
            x,y = p
            x1 = x - (x % 10)
            x2 = x1 + 10

            y1 = y - (y % 10)
            y2 = y1 + 10

            draw.rectangle([(x1, y1), (x2, y2)], fill=rgb)
        
        image = image.crop(area)

        plt.figure(figsize=(20,10))
        plt.axis('off')
        plt.imshow(image)
        plt.savefig(draw_path + 'test_{0:05d}.png'.format(i))
        plt.close()
        
        del draw


# fps = 총 좌표 개수 / 영상 길이 
def img2avi(fps):

    array2img()
    img_array = []

    print('------ image to avi file ... ----------')
    for filename in tqdm(glob.glob(os.getcwd() + '/pillow_img/*.png')):
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    out = cv2.VideoWriter('test.avi', cv2.VideoWriter_fourcc(*'DIVX'), int(fps), size)

    for i in tqdm(img_array):
        out.write(i)
    out.release()


# fps 설정 => 총 좌표 개수 / 영상 길이
img2avi(60)