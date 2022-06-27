import cv2
import mediapipe as mp
import numpy as np
import time
import os
import glob
from tqdm import tqdm
import convert as cvt
import hand_draw_extract as hde

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# 영상 파일 경로 설정
filepath = './test.mp4'

# For webcam input:
cap = cv2.VideoCapture(filepath)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
writer = cv2.VideoWriter('test_sv.avi', fourcc, 24, (int(width), int(height)))

frame = []

hde.hand_draw_extract()
print(len(frame))

# 처음 새 폴더를 임의 생성, 이후 주석 처리
# os.mkdir('./test_img')

for idx in tqdm(range(len(frame)-1)):
    cvt.array_to_tuple(idx)
    cvt.draw_sav_img(idx)

cvt.create_video()
