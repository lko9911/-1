 # 소프트웨어 알고리즘 순서도 🧮
![image01](https://github.com/user-attachments/assets/fb4a1854-9cc7-4450-94fd-cafe740f38d3)


## 1단계. RGB 카메라 인식 ✔

▷ 라즈베리파이5와 연동되는 RGB 카메라 (케이블 확인) 혹은 스테레오 카메라, 스테레오 카메라 웹캠도 가능 (USB 연결)<br>
▷ 화질개선을 위한 영상 전처리 작업 (컴퓨터 비전 사용이 가능하므로 문제 없음)
<br><br>

## 2단계. VNC server or Internal device ✔

▷ VNC server는 원격 제어 (노트북으로 전체 관리), Internal device는 별도의 터치스크린과 키보드로 구성해 로봇 자체에서 제어함<br>
▷ VNC server는 putty로 와이파이 연동으로 제어 (SSH 설정), intetnal device는 별도의 장비를 구비 해야 함 
<br>
### - Stereo vision
▷ 캘리브레이션 (체스보드 분석) → 캘리브레이션 파일을 통한 좌/우 영상의 보정 (정합 이미지 생성) → 깊이 사진으로 표현<br>
### - YOLO Detection
좌/우 사진에서 YOLO Detection 수행 : 바운딩 박스 중심 좌표 → 캘리브레이션 파일을 통한 좌/우 영상의 보정 (정합 이미지 생성) → 깊이 사진에서 바운딩 박스 중심 좌표의 깊이 계산
<br><br>
## 3단계 Robot control ✔

▷ 라즈베리파이5에서 모터 제어 예정
