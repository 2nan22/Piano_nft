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

  mp_drawing = mp.solutions.drawing_utils
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

  writer = cv2.VideoWriter(after, fourcc, int(video_fps), (int(width), int(height)))

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

# 받아온 프레임 별 좌표 값(list)를 각각 csv 파일로 저장
def list2csv(mc, mdc, mtc):
    
    frame_l, frame_r = hand_draw_extract(mc, mdc, mtc)

    now_time = datetime.now()
    current_time = now_time.strftime('%m%d_%H%M')

    file_lst = './point_csv/'

    if not os.path.exists(file_lst):
        os.mkdir(file_lst)

    with open(file_lst + current_time  + '_' + filepath[:-4] + '_left.csv','w') as file :

      write = csv.writer(file)
      write.writerows(frame_l)

    with open(file_lst  + current_time  + '_' + filepath[:-4] + '_right.csv','w') as file :

      write = csv.writer(file)
      write.writerows(frame_r)

    return (current_time + '_' + filepath[:-4] + '_left.csv',
            current_time + '_' + filepath[:-4] + '_right.csv')


# columns name 매핑하기 위한 함수
# 0은 left / 1은 right 
# 하나의 json 파일에 담기 전, 각각 나누기 위해 이를 분류
def mapping_col(side):

  if side == 0:
    side = '_l'
  elif side == 1:
    side = '_r'

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

  dic_key = [c + side for c in dic_key]

  return dic_key

# csv 파일을 mapping 하고, 하나로 합쳐 csv / json 파일로 송출
def csv2json(mc, mdc, mtc, json_=filepath[:-4]):

    # csv 파일 이름 가져오기
    csv_left, csv_right = list2csv(mc, mdc, mtc)
    
    # 저장 이름에 담을 시간 정보 생성
    now_time = datetime.now()
    current_time = now_time.strftime('%m%d_%H%M')

    # csv 저장 경로, json 저장 경로
    csv_path = './point_csv/'
    file_lst = './point_json/'

    if not os.path.exists(file_lst):
        os.mkdir(file_lst)

    # json 파일 저장할 path, name
    json_path = file_lst  + current_time + '_' + json_  + '.json'

    # mapping 을 위한 변수 생성
    key_l = mapping_col(0)
    key_r = mapping_col(1)

    # 각 손의 좌표 값 불러오기
    point_l = pd.read_csv(csv_path + csv_left, names=key_l)
    point_r = pd.read_csv(csv_path + csv_right, names=key_r)


    # 리스트 형태로 담겨 있는 값들을 각각 x, y, z 를 key 값으로 dict 형태로 구성
    dic_list = ['x', 'y', 'z']

    for k in key_l:
      point_l[k] = point_l[k].apply(lambda x : {i:j for i,j in zip(dic_list, literal_eval(x))}) 

    for k in key_r:
      point_r[k] = point_r[k].apply(lambda x : {i:j for i,j in zip(dic_list, literal_eval(x))}) 

    # 병합 후 결측치는 0으로 대체, csv파일로 저장
    df = pd.concat([point_l, point_r], axis=1)
    df.fillna(0, inplace=True)
    df.to_csv(csv_path + current_time + '_' + filepath[:-4] + '_full.csv', index=False)

    # json 파일로 저장
    jso = df.to_json(orient='index')
    parsed = json.loads(jso)

    with open(json_path, 'w') as f:
        json.dump(parsed, f, indent=4)

################# 설정 가능 ##########################
# 1) 모델 복잡도 / 2) 모델 감지 신뢰도 / 3) 모델 추적 신뢰도
#    0  or  1           0  ~  1              0  ~  1
# 모델 복잡도를 높이면 더 정교하게 추적, 감지가 가능하지만, 성능 / 프레임 측면에서 힘들어함
# 감지, 추적 신뢰도는 높일수록 신뢰도가 높아지지만, 추적 / 감지가 보다 잘 되지 않음
# 현재는 model의 default 값 사용. 조절 가능

csv2json(0, 0.5, 0.5)
