import cv2
import mediapipe as mp
import numpy as np
import time
from tqdm import tqdm

# global filepath

print('----------------------------------------------------------------------------------')
filepath = input('Type the test file name (확장자명 제외)) : ') + '.mp4'
print('----------------------------------------------------------------------------------')
model_complex = input('모델 복잡도(0 or 1) 숫자를 입력하세요 ' + '\n' + '(default = 1)' + ' : ')
print('----------------------------------------------------------------------------------')
min_detect = input('최소 감지 신뢰도(0 ~ 1) 숫자를 입력하세요 ' + '\n' + '(default = 0.5)' + ' : ')
print('----------------------------------------------------------------------------------')
min_tracking = input('최소 추적 신뢰도(0 ~ 1) 숫자를 입력하세요' + '\n' + '(default = 0.5)' +  ' : ')


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(filepath)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
video_fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
writer = cv2.VideoWriter('after_mp.avi', fourcc, int(video_fps), (int(width), int(height)))

frame_l = []
frame_r = []


def hand_draw_extract(mc, mdc, mtc):
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
            
            # Iterate over the found hands.
            for hand_no, hand_landmarks in enumerate(results.multi_hand_landmarks):
        
                if hand_no == 0:
                  for i in range(21):
                      frame_per_vector_l.append(np.array([
                                    hand_landmarks.landmark[mp_hands.HandLandmark(i).value].x, 
                                    hand_landmarks.landmark[mp_hands.HandLandmark(i).value].y,
                                    # results.multi_hand_landmarks[i].landmark[v].z,
                                ]))
                  frame_l.append(np.array(frame_per_vector_l))
                elif hand_no == 1:
                  for i in range(21):
                      frame_per_vector_r.append(np.array([
                                    hand_landmarks.landmark[mp_hands.HandLandmark(i).value].x, 
                                    hand_landmarks.landmark[mp_hands.HandLandmark(i).value].y,
                                    # results.multi_hand_landmarks[i].landmark[v].z,
                                ]))
                  frame_r.append(np.array(frame_per_vector_r))
                

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
  
  return frame_l, frame_r