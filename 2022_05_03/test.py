import cv2
import mediapipe as mp
import numpy as np
import time
import os

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

def hand_frames():
  # Mediapipe 객체 생성
  with mp_hands.Hands(
      model_complexity=0, # 복잡도 => 정확도 및 추론 지연 시간과 상관관계
      min_detection_confidence=0.5, # 최소 감지 신뢰도
      min_tracking_confidence=0.5 # 최소 추적 신뢰도
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
            # results.multi_hand_landmarks[0] : left / [1] : right
            for i in range(len(results.multi_hand_landmarks)):
              # results.multi_hand_landmarks[i].landmark[v] => v는 각 skeleton point (21개)
              for v in range(len(results.multi_hand_landmarks[i].landmark)):
                # hand tracking이 제대로 되지 않을 경우, 해당 프레임의 값은 nan으로 처리

                # 한 손당 정확도가 최소 0.9 이상 정도 되면 값 추가
                # hand_type = 
                print(results.multi_handedness)
                if results.multi_handedness[i].classification[0].score >= 0.9:
                  frame_per_vector.append(np.array([
                      results.multi_hand_landmarks[i].landmark[v].x * width, # mediapipe 내에서 너비, 높이 값으로 정규화한 값을 좌표 값으로 만들었기 때문에 다시 풀어주기
                      results.multi_hand_landmarks[i].landmark[v].y * height,
                      results.multi_hand_landmarks[i].landmark[v].z,
                  ]))
                else:
                  frame_per_vector.append(np.array([np.nan, np.nan, np.nan]))
              frame.append(np.array(frame_per_vector))

            for num, hand_landmarks in enumerate(results.multi_hand_landmarks):

                mp_drawing.draw_landmarks(image,
                                          hand_landmarks, 
                                          mp_hands.HAND_CONNECTIONS,  # 라인 연결을 위한 값(튜플 형태)
                                          mp_drawing.DrawingSpec(color=(255, 229, 180), thickness=2, circle_radius=2),
                                          mp_drawing.DrawingSpec(color=(255, 229, 180), thickness=2, circle_radius=1),)
                                          # mp_drawing_styles.get_default_hand_landmarks_style(),  # 포인트 drawing default 값
                                          # mp_drawing_styles.get_default_hand_connections_style()) # 라인 drawing default 값
              
            
          elapse_time = time.time() - start_time
          fps = 1 / elapse_time

          cv2.putText(image, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))
          writer.write(image)          
          cv2.imshow('MediaPipe Hands', image)
          
          if cv2.waitKey(5) & 0xFF == ord('q'):
            break
      else:
        break  
    cap.release()
    writer.release()
    cv2.destroyAllWindows()
