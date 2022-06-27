import cv2
import mediapipe as mp
import numpy as np
import time
from tqdm import tqdm

# global filepath
filepath = './test_' + input('Type the test file name (test_ 이후 문자만)) : ') + '.mp4'

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(filepath)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
video_fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
writer = cv2.VideoWriter('after_mp.avi', fourcc, int(video_fps), (int(width), int(height)))

frame = []

def hand_draw_extract(mdc, mtc):
  # Mediapipe 객체 생성
  with mp_hands.Hands(
      model_complexity=0, # 복잡도 => 정확도 및 추론 지연 시간과 상관관계
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
          
          # image = cv2.flip(image, 1)
          
          # Make detection
          results = hands.process(image)
          
          # Recolor back to BGR
          image.flags.writeable = True
          image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
          
          if results.multi_hand_landmarks:
            
            # 좌표 값 추출 후 array 형태(x, y, z)로 list에 append
            frame_per_vector = []
            try: 
              for i in range(2):
                if results.multi_hand_landmarks[i] == None:
                  frame_per_vector.append(np.zeros((21, 2)))
                else:
                  for v in range(len(results.multi_hand_landmarks[i].landmark)): 
                    frame_per_vector.append(np.array([
                                  results.multi_hand_landmarks[i].landmark[v].x, 
                                  results.multi_hand_landmarks[i].landmark[v].y
                                  # results.multi_hand_landmarks[i].landmark[v].z,
                              ])) 

                frame.append(np.array(frame_per_vector))
            except:
              continue


            for num, hand_landmarks in enumerate(results.multi_hand_landmarks):

                mp_drawing.draw_landmarks(image,
                                          hand_landmarks, 
                                          mp_hands.HAND_CONNECTIONS,  # 라인 연결을 위한 값(튜플 형태)
                                          mp_drawing_styles.get_default_hand_landmarks_style(),  # 포인트 drawing default 값
                                          mp_drawing_styles.get_default_hand_connections_style()) # 라인 drawing default 값
                                          # mp_drawing.DrawingSpec(color=(255, 229, 180), thickness=2, circle_radius=2),
                                          # mp_drawing.DrawingSpec(color=(255, 229, 180), thickness=2, circle_radius=1),)
              
            
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
  
  return frame