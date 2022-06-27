import os
import numpy as np
import pandas as pd
import cv2
import mediapipe as mp
import time
from datetime import datetime
import csv
import json
from ast import literal_eval

filepath = input('Type the test file name (확장자 포함 full name) : ')

def hand_draw_extract(mc, mdc, mtc):

  mp_drawing = mp.solutions.drawing_utilsSA
  mp_drawing_styles = mp.solutions.drawing_styles
  mp_hands = mp.solutions.hands

  cap = cv2.VideoCapture(filepath)
  width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
  height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
  video_fps = cap.get(cv2.CAP_PROP_FPS)
  fourcc = cv2.VideoWriter_fourcc(*'XVID')
  
  after = './after_mp.avi'

  if os.path.exists(after):
      os.remove(after)

  writer = cv2.VideoWriter('after_mp.avi', fourcc, int(video_fps), (int(width), int(height)))

  frame_l = []
  frame_r = []

  # Mediapipe 객체 생성
  with mp_hands.Hands(
      model_complexity=mc, # 복잡도 => 정확도 및 추론 지연 시간과 상관관계
      min_detection_confidence=mdc, # 최소 감지 신뢰도
      min_tracking_confidence=mtc # 최소 추적 신뢰도
      ) as hands:
    
    while cap.isOpened():
    
      success, image = cap.read()
      start_time = time.time()
    
      if success: 
          # Recolor image
          image.flags.writeable = False
          image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
          
          # Make detection
          results = hands.process(image)
          
          # Recolor back to BGR
          image.flags.writeable = True
          image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
          
          if results.multi_hand_landmarks:
            
            # 좌표 값 추출 후 array 형태(x, y, z)로 list에 append
            frame_per_vector_l = []
            frame_per_vector_r = []
            
            for hand_no, hand_landmarks in enumerate(results.multi_hand_landmarks):
                
                if hand_no == 0:
                  for i in range(21):
                      frame_per_vector_l.append([hand_landmarks.landmark[mp_hands.HandLandmark(i).value].x,
                                                hand_landmarks.landmark[mp_hands.HandLandmark(i).value].y,
                                                hand_landmarks.landmark[mp_hands.HandLandmark(i).value].z])
                                                
                  frame_l.append(frame_per_vector_l)

                elif hand_no == 1:
                  for i in range(21):
                      frame_per_vector_r.append([hand_landmarks.landmark[mp_hands.HandLandmark(i).value].x,
                                                hand_landmarks.landmark[mp_hands.HandLandmark(i).value].y,
                                                hand_landmarks.landmark[mp_hands.HandLandmark(i).value].z])
                                                
                  frame_r.append(frame_per_vector_r)

                mp_drawing.draw_landmarks(image,
                                          hand_landmarks, 
                                          mp_hands.HAND_CONNECTIONS,  # 라인 연결을 위한 값(튜플 형태)
                                          mp_drawing_styles.get_default_hand_landmarks_style(),  # 포인트 drawing default 값
                                          mp_drawing_styles.get_default_hand_connections_style()) # 라인 drawing default 값


          elapse_time = time.time() - start_time
          fps = 1 / elapse_time

          cv2.putText(image, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))
          cv2.putText(image, f'width: {int(width)}', (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))
          cv2.putText(image, f'height: {int(height)}', (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))

          writer.write(image)          
          cv2.imshow('MediaPipe Hands', image)
          
          if cv2.waitKey(5) & 0xFF == ord('q'):
            break
      else:
        break  
  cap.release()
  writer.release()
  cv2.destroyAllWindows()
  
  return frame_l, frame_r


def array2csv(mc, mdc, mtc):
    
    frame_l, frame_r = hand_draw_extract(mc, mdc, mtc)

    now_time = datetime.now()
    current_time = now_time.strftime('%m%d_%H%M')

    file_lst = './point_csv'

    if not os.path.exists(file_lst):
        os.mkdir(file_lst)

    with open(file_lst + '/' + filepath[:-4] + '_left_' + current_time + '.csv','w') as file :

      write = csv.writer(file)
      write.writerows(frame_l)

    with open(file_lst + '/' + filepath[:-4] + '_right_'  + current_time + '.csv','w') as file :

      write = csv.writer(file)
      write.writerows(frame_r)

    return (filepath[:-4] + '_left_' + current_time + '.csv',
            filepath[:-4] + '_right_'  + current_time + '.csv')

def mapping_col():

  label_dict = {  
    'WRIST' : 0,
    'THUMB_CMC' : 1,
    'THUMB_MCP' : 2,
    'THUMB_IP' : 3,
    'THUMB_TIP' : 4,
    'INDEX_FINGER_MCP' : 5,
    'INDEX_FINGER_PIP' : 6,
    'INDEX_FINGER_DIP' : 7,
    'INDEX_FINGER_TIP' : 8,
    'MIDDLE_FINGER_MCP' : 9,
    'MIDDLE_FINGER_PIP' : 10,
    'MIDDLE_FINGER_DIP' : 11,
    'MIDDLE_FINGER_TIP' : 12,
    'RING_FINGER_MCP' : 13,
    'RING_FINGER_PIP' : 14,
    'RING_FINGER_DIP' : 15,
    'RING_FINGER_TIP' : 16,
    'PINKY_MCP' : 17,
    "PINKY_PIP" : 18,
    'PINKY_DIP' : 19,
    'PINKY_TIP' : 20
  }

  dic_key = list(label_dict.keys())

  return dic_key

def csv2json(mc, mdc, mtc, json_=filepath[:-4]):

    csv_left, csv_right = array2csv(mc, mdc, mtc)
    
    now_time = datetime.now()
    current_time = now_time.strftime('%m%d_%H%M')

    csv_path = './point_csv/'

    file_lst = './point_json/'

    if not os.path.exists(file_lst):
        os.mkdir(file_lst)

    json_path_l = file_lst + json_ + '_left_' + current_time + '.json'
    json_path_r = file_lst + json_ + '_right_' + current_time + '.json'


    key = mapping_col()
    point_l = pd.read_csv(csv_path + csv_left, names=key)
    point_r = pd.read_csv(csv_path + csv_right, names=key)

    for k in key:
        point_l[k] = point_l[k].apply(lambda x : literal_eval(x))

    for k in key:
        point_r[k] = point_r[k].apply(lambda x : literal_eval(x))


    jso_l = point_l.to_json(orient='index')
    jso_r = point_r.to_json(orient='index')
    
    parsed_l = json.loads(jso_l)
    parsed_r = json.loads(jso_r)

    with open(json_path_l, 'w') as f:
        json.dump(parsed_l, f, indent=4)

    with open(json_path_r, 'w') as f:
        json.dump(parsed_r, f, indent=4)

# 모델 복잡도 / 모델 감지 신뢰도 / 모델 추적 신뢰도 / 저장할 json name
csv2json(0, 0.5, 0.5)
