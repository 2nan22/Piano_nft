## 2022_05_03

### - 좌표 값 추출 함수 생성
### - hand tracking이 되지 않을 때, 해당 값을 nan 값으로 대체하는 조건문을 보완해야함
### - 좌표 값에 대해 array 형태를 어떻게 만들어야 될 지 고민해야함

#### handlandmark_keypoints
<br>


![hand_landmarks](https://user-images.githubusercontent.com/75026887/166424498-a1f2d729-fa4c-4383-80c7-37ba0047b4d8.png)

<br><br><br>


## 2022_05_04

### - 좌표 값 추출 시 issue fix
### - 양 손을 하나의 리스트에 다 담았기 때문에, 번갈아가면서 인덱싱이 되도록 하려고   
###   비는 공간은 Nan 값을 채우려고 했는데 잘 되지 않음
### - 간단히 뽑아진 좌표 값 test를 위해 cv2로 line을 그리고 img를 영상화해서 test
###   - 한 손만 표현이 됨. 알고리즘 수정이 필요.... 어디를 수정.. .. ..
 
- Test용 영상
https://drive.google.com/file/d/1BJzv-HFcA7bFroIDDBpuHr415kp2L15u/view?usp=sharing, 
- Test 이후 Output 영상
https://drive.google.com/file/d/1lJXoBLhNxmlV_hSk_cEAHVXTxjDtNkq8/view?usp=sharing

### - 양 손이 함께 표현되는 로직을 조금 더 고민해야함
### - 어떻게 표현할 지 고민
#### 1) 리듬 게임 형식
#### 2) 파동, 그래프 형식
#### 3) 파동, (잔잔한 물에 물방울이 떨어지는 느낌을 어떻게 주지)


## 2022_05_11

### 수정)

##### 1) test.py 실행 시, 테스트 할 파일 이름( ex) test_side.mp4 => 입력 시 side만 입력) 입력
##### 최소 감지 / 추적 신뢰도 입력하게 변경
##### 2) 최종 영상 생성은 해당 함수 실행 시간으로 생성 (ex) 0511_1254)
##### 3) 흰 배경에 skeleton만 표현.
##### => 이를 audio visual과 합쳐서 시각화하면 dynamic한 visual을 제공할 수 있지 않을까?
##### => skeleton의 표현 방식이 조금 더 고급스럽게 나올 수 있는 라이브러리나 js를 사용하는 방안을 강구 중
##### - skeleton만으로는 단조로울 수 있고, 한 손만 표출되는 문제점을 audio를 시각화해서 함께 표현할 수 있는 형식을 고민


### Audio Visualization 예제 코드 탐색)

##### 1) pygame audio visualization
- 음악 흐름에 맞추어 음역대에 따라 흐름을 시각화
- 예제 코드의 경우 원을 둘러싼 막대들이 움직이는 형상.
- 이를 파동 그래프 형태로 바꾸어 파도의 움직임을 형상화하는 느낌을 주고 싶은데,
 그래프의 형태로 표현하려면 해당 코드의 로직을 대거 수정해야 하는 부분
- 예제 코드 첨부
##### 2) Blender
- 3D 렌더링 툴을 이용해서 직접 그려보려 했으나,
 사용법이 복잡하여 단시간 내 작업물 산출이 어려울 것으로 판단
- 고사양이 아니라면, 사용이 어려움
영상 :
##### 3) Fractal 형태
- 현대 미술과 가까운 형태로 표현하는 것이 좋은지..
- 이것이 진정 무슨 의미를 내포하는 지는 모르겠다. 단지 시각화하는 데 있어 하나의 기법 정도
로만 생각. 의미를 담으려니 떠오르는게 없다
- 예제
영상 : https://www.youtube.com/watch?v=af16HnHCSTA
깃허브 코드 :
https://github.com/m-bartlett/python-fun/blob/master/LoopingMultithreadNewton.py
##### 4) pyaudioFFT
- audio 소리를 flow 형태로 만들어줌
- 개인적으로 이러한 형태로 색 조정 및 스켈레톤을 함께 보여주면 괜찮겠다는 생각
영상 : https://www.youtube.com/watch?v=FnP2bkzU4oo
깃허브 코드 : https://github.com/aiXander/Realtime_PyAudio_FFT

***
skeleton과 audio visualization 된 영상이 따로 나오면 이를 자연스럽게 합칠 수 있는가?
영상 + 영상이 자연스럽게 되는가에 대해 고민
### Idea sketch
##### - 서혜경 님이 관객을 고려해 베토벤의 전곡으로 무대를 구성,
##### - 서혜경 님의 인터뷰를 바탕으로 본 결과, Romantic / 부드러움의 keyword가 내포되어 있음
##### - 공연하실 베토벤의 곡들을 쭉 들어본 결과, 음악에 문외한이지만 강약 조절에 Point가 있다고 느낌
##### => 마치 인생의 성장기와 같은 느낌을 주었다. 잔잔할 때는 잔잔하다가, 거친 파도를 맞설 때는 동적인 그러한 느낌을 받았다
##### 1) skeleton + 파형 그래프
- 인생은 망망대해에 떠 있는 단배와 같다라고 생각. 잔잔한 파도, 혹은 거친 파도에 맞서 계속해서 나아가는 파형 그래프를 통해 파도를 표현하고, skeleton을 함께 표현하여 무언가 선지자? 조력자? 와 같 은 느낌을 통해 나아가는 느낌
##### 2) skeleton + 나무 형상(?)
- 인생으로 keyword를 잡아서 성장하는 나무가 떠올랐다. 나무가 점점 자라고, 많은 가지들을 만들어내고 이와 함께 skeleton을 표현하면 마치 성장을 돕는 조력자. 어머니와 같은 느낌을 주지 않을까라는 생각이 들었다.
##### 3) skeleton + 원 형태에 audio visualization
- code를 파악하고 customizing하면 괜찮을 수 있겠다 생각.


## 2022_06_20

### 수정)

#### 1) 함수 구조 수정

- 1개의 파일에서 전부 실행되게끔, line drawing 후 저장하는 코드 지우고 단순히 영상에서 좌표 값을 추출하는 코드 저장(csv / json)까지만 구현

#### 2) 좌표 값 오류 제거

- 뽑아 놓은 좌표 값을 처음에 dictionary 구조로 하나씩 뽑았더니, 마지막 값으로 전부 중복되는 문제 발생, 이를 나중에 발견
- list 형태로 append 후, csv 파일로 저장
- 이후 csv 파일 불러와서 좌표 부위 별로 column을 mapping하고 프레임 / 부위별로 json 형태의 파일 생성
