# 📖 스마트폰을 사용한 스테레오 비전 연구 결과 

### 캘리브레이션 : 아주 조금의 오차에도 크게 영향받는 것을 확인
![left_with_corners_l2 (4)](https://github.com/user-attachments/assets/91555619-4357-4adc-a9f6-4b80d2ccc3a8) (좌측)
![right_with_corners_l2 (4)](https://github.com/user-attachments/assets/fb783cdb-677e-4b79-b28b-e45f07f0e475) (우측)

### 깊이 분석 : 정렬, 정합 깊이 사진 확인 
![rectified_left](https://github.com/user-attachments/assets/8cf293b6-8719-4eb2-bfe4-0b408378a6f7)
(좌측 정렬 사진)
![rectified_right](https://github.com/user-attachments/assets/2805055d-8877-4c31-8d90-90fa312f29d6)
(우측 정렬 사진)
→ 캘리브레이션 데이터 R의 값이 오차가 생겼음을 알 수 있음

![disparity](https://github.com/user-attachments/assets/a5692a9c-9aa7-400c-8733-fc8f408780cd)
(깊이 사진 분석 결과 : 특정 위치의 깊이는 얼추 맞기는 함)

![overlay_corrected_image](https://github.com/user-attachments/assets/d8ff7941-bea3-498c-b5a7-b221f7b93b87)
(보정된 오버레이 정합 사진 : 정렬 제외)
→ 정합에 대한 오차가 발생

![스크린샷 2024-11-21 130016](https://github.com/user-attachments/assets/9d121cf9-b3c5-47ac-882b-2718c8e8af1c) (좌측 매칭점및 연결선)
![스크린샷 2024-11-21 130031](https://github.com/user-attachments/assets/9ed42ae1-429c-41ac-bcc1-1fc818feea19) (우측 매칭점및 연결선)
![스크린샷 2024-11-21 130047](https://github.com/user-attachments/assets/872ebd91-b691-4f89-9237-bae4b3f0de00) (깊이 사진)
![스크린샷 2024-11-21 130102](https://github.com/user-attachments/assets/5e1132a4-850f-4942-8f69-b8443d33e7b3) <br>(연결선 시각화)
→ 좌/우 사진의 기본적인 오차로 정합이 올바르지 않다는 걸 알아냄
→ 정합이 완벽한 사진에 대한 깊이 분석 성공 확인
<br><br>
### 결론 : 카메라 2개 보다는 스테레오 카메라가 더욱 좋다. 
