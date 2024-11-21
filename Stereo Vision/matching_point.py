import cv2
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

# Step 1: 캘리브레이션 데이터 로드
calib_data = np.load("stereo_calibration_data.npz")
mtx_left = calib_data['camera_matrix_left']
dist_left = calib_data['dist_coeffs_left']
mtx_right = calib_data['camera_matrix_right']
dist_right = calib_data['dist_coeffs_right']
F = calib_data['fundamental_matrix']

# Step 2: 스테레오 이미지 로드
img_left = cv2.imread('left_sample2.png')
img_right = cv2.imread('right_sample2.png')

# 렌즈 왜곡 제거를 하지 않음 (원본 이미지 그대로 사용)

# Step 3: 에피폴라 선 시각화를 위한 함수
def draw_epilines(img1, img2, lines, pts1, pts2):
    """
    에피폴라 선을 그리는 함수
    img1: 선을 그릴 이미지 (좌측 또는 우측)
    img2: 대응점만 표시할 이미지 (우측 또는 좌측)
    lines: 에피폴라 선 방정식
    pts1, pts2: 매칭된 특징점
    """
    r, c = img1.shape[:2]
    img1 = img1.copy()
    img2 = img2.copy()
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2] / r[1]])  # x=0에서의 y
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])  # x=cols에서의 y
        img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)
        img1 = cv2.circle(img1, tuple(pt1), 5, color, -1)
        img2 = cv2.circle(img2, tuple(pt2), 5, color, -1)
    return img1, img2

# Step 4: 특징점 추출 및 매칭
sift = cv2.SIFT_create()
keypoints1, descriptors1 = sift.detectAndCompute(img_left, None)
keypoints2, descriptors2 = sift.detectAndCompute(img_right, None)

bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
matches = bf.knnMatch(descriptors1, descriptors2, k=2)

# 좋은 매칭만 선택 (Lowe's ratio test)
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

# 매칭 점 추출
pts1 = np.int32([keypoints1[m.queryIdx].pt for m in good_matches])
pts2 = np.int32([keypoints2[m.trainIdx].pt for m in good_matches])

# RANSAC으로 정합성 필터링
F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)
pts1 = pts1[mask.ravel() == 1]
pts2 = pts2[mask.ravel() == 1]

# Step 5: 에피폴라 선 계산
lines_left = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F).reshape(-1, 3)
lines_right = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F).reshape(-1, 3)

# Step 6: 시각화
img1_epilines, img2_points = draw_epilines(img_left, img_right, lines_left, pts1, pts2)
img2_epilines, img1_points = draw_epilines(img_right, img_left, lines_right, pts2, pts1)

# Step 7: 결과 저장 및 출력
cv2.imwrite("left_with_epilines.jpg", img1_epilines)
cv2.imwrite("right_with_epilines.jpg", img2_epilines)

cv2.imshow("Left Image with Epilines", img1_epilines)
cv2.imshow("Right Image with Epilines", img2_epilines)

cv2.waitKey(0)
cv2.destroyAllWindows()

# Step 8: 디스패리티 맵 계산을 위한 스테레오 매칭 설정
# 스테레오 사물의 매칭을 위한 StereoBM(블록 매칭)
stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
disparity_map = stereo.compute(cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY),
                               cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY))

# 디스패리티 맵 결과를 시각화
disparity_map_normalized = cv2.normalize(disparity_map, None, 0, 255, cv2.NORM_MINMAX)
disparity_map_normalized = np.uint8(disparity_map_normalized)

# Step 9: 결과 저장 및 출력
cv2.imwrite("disparity_map.jpg", disparity_map_normalized)
cv2.imshow("Disparity Map", disparity_map_normalized)
cv2.waitKey(0)

# Step 10: 3D 포인트 재구성
# 카메라 매트릭스 및 왜곡 계수로 3D 포인트를 재구성
Q = np.array([[1, 0, 0, -0.5 * img_left.shape[1]],  # Q 행렬 계산
              [0, -1, 0, 0.5 * img_left.shape[0]],
              [0, 0, 0, -mtx_left[0, 0]],  # 카메라의 초점 거리
              [0, 0, 1, 0]])

# 스테레오 매칭된 점들로 3D 재구성
points_3d = cv2.reprojectImageTo3D(disparity_map, Q)

# 3D 점 중 일부를 선택하여 시각화
mask = disparity_map > disparity_map.min()
out_points = points_3d[mask]
'''
# 3D 포인트 시각화 (Open3D 또는 matplotlib을 사용할 수 있음)
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(out_points)
o3d.visualization.draw_geometries([pcd])
'''

# Step 11: 특징점 매칭 결과 시각화
def visualize_keypoints(img1, img2, keypoints1, keypoints2, good_matches):
    """
    특징점 매칭 결과를 시각화하는 함수
    img1, img2: 비교할 이미지
    keypoints1, keypoints2: 두 이미지의 특징점들
    good_matches: 좋은 매칭을 나타내는 매칭 리스트
    """
    # 매칭된 점들을 그리기 위한 준비
    img_matches = cv2.drawMatches(img_left, keypoints1, img_right, keypoints2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    # Matplotlib을 사용하여 비교 이미지 출력
    plt.figure(figsize=(12, 6))
    plt.imshow(img_matches)
    plt.title('Matched Keypoints Between Left and Right Images')
    plt.axis('off')  # 축 숨기기
    plt.show()

# Step 12: 특징점 매칭 결과 시각화 호출
visualize_keypoints(img_left, img_right, keypoints1, keypoints2, good_matches)
