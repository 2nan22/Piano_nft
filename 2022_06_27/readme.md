- Array to image

좌표 array 값을 pillow로 평면에 그리고, 뒷 배경이 그림을 이어 붙였을 때 움직이는 것처럼 표현하기 위해서

프레임 별로 변화하는 알고리즘을 생성

- image to avi

cv2를 이용해서 만들어진 이미지들을 avi 파일로 변환.
영상 중간 중간 첫 화면이 중첩되어 나타나서 화면이 튀는 것처럼 보임
cv2 이외 ffmpeg 등 다른 라이브러리를 사용해서 issue를 해결할 예정