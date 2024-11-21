import cv2
import numpy as np
import glob
import os
# rich 사용 예정 

# 체스보드 패턴 크기
chessboard_size = (7, 7)
square_size = 0.03  # 체스보드 사각형의 크기 (단위: 미터)

# 3D 체스보드 좌표 생성
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size

# 이미지에서 추출한 객체 및 이미지 포인트
objpoints = []  # 3D 점
imgpoints_left = []  # 좌측 카메라 2D 점
imgpoints_right = []  # 우측 카메라 2D 점

# 좌우 카메라 이미지 경로
images_left = sorted(glob.glob('left/*.jpg'))
images_right = sorted(glob.glob('right/*.jpg'))

# 보정 기준
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 체스보드 코너 찾기
for img_left, img_right in zip(images_left, images_right):
    img_l = cv2.imread(img_left)
    img_r = cv2.imread(img_right)
    gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

    ret_l, corners_l = cv2.findChessboardCorners(gray_l, chessboard_size, None)
    ret_r, corners_r = cv2.findChessboardCorners(gray_r, chessboard_size, None)

    if ret_l and ret_r:
        print(f"체스보드 코너 찾음: {img_left}, {img_right}")
        objpoints.append(objp)

        corners2_l = cv2.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), criteria)
        corners2_r = cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1), criteria)

        imgpoints_left.append(corners2_l)
        imgpoints_right.append(corners2_r)

        # 체스보드 코너 그리기 (좌측, 우측 이미지에)
        img_l = cv2.drawChessboardCorners(img_l, chessboard_size, corners2_l, ret_l)
        img_r = cv2.drawChessboardCorners(img_r, chessboard_size, corners2_r, ret_r)

        # 저장할 폴더가 존재하는지 확인하고 없으면 생성
        output_dir = "result"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 저장 경로 확인
        left_output_path = os.path.join(output_dir, f"left_with_corners_{os.path.basename(img_left)}")
        right_output_path = os.path.join(output_dir, f"right_with_corners_{os.path.basename(img_right)}")
        
        '''
        # 파일 저장
        if cv2.imwrite(left_output_path, img_l):
            print(f"파일 저장 성공: {left_output_path}")
        else:
            print(f"파일 저장 실패: {left_output_path}")

        if cv2.imwrite(right_output_path, img_r):
            print(f"파일 저장 성공: {right_output_path}")
        else:
            print(f"파일 저장 실패: {right_output_path}")
        '''
        
        # 이미지 출력 (체스보드 코너가 그려진 상태)
        cv2.imshow('Left Chessboard Corners', img_l)
        cv2.imshow('Right Chessboard Corners', img_r)
        cv2.waitKey(500)  # 잠시 대기 후 계속

    else:
        print(f"체스보드 코너를 찾지 못함: {img_left}, {img_right}")

cv2.destroyAllWindows()

# 좌우 카메라 개별 캘리브레이션
_, mtx_left, dist_left, _, _ = cv2.calibrateCamera(objpoints, imgpoints_left, gray_l.shape[::-1], None, None)
_, mtx_right, dist_right, _, _ = cv2.calibrateCamera(objpoints, imgpoints_right, gray_r.shape[::-1], None, None)

# 스테레오 캘리브레이션
flags = cv2.CALIB_FIX_INTRINSIC
criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
_, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
    objpoints, imgpoints_left, imgpoints_right, 
    mtx_left, dist_left, mtx_right, dist_right, 
    gray_l.shape[::-1], criteria=criteria_stereo, flags=flags
)

# 결과 저장
calib_data = {
    "camera_matrix_left": mtx_left,
    "dist_coeffs_left": dist_left,
    "camera_matrix_right": mtx_right,
    "dist_coeffs_right": dist_right,
    "rotation_matrix": R,
    "translation_vector": T,
    "essential_matrix": E,
    "fundamental_matrix": F
}

np.savez("stereo_calibration_data.npz", **calib_data)

print("스테레오 캘리브레이션 완료! 결과는 'stereo_calibration_data.npz'에 저장되었습니다.")


# 두 이미지를 로드 (보정된 이미지 시각화)
img_left = cv2.imread('left_sample.jpg')
img_right = cv2.imread('right_sample.jpg')

# 스테레오 이미지 정합
R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
    mtx_left, dist_left, mtx_right, dist_right, gray_l.shape[::-1], R, T
)

# 정합된 프로젝션 행렬
map1_left, map2_left = cv2.initUndistortRectifyMap(mtx_left, dist_left, R1, P1, gray_l.shape[::-1], cv2.CV_32F)
map1_right, map2_right = cv2.initUndistortRectifyMap(mtx_right, dist_right, R2, P2, gray_r.shape[::-1], cv2.CV_32F)

# 좌/우 이미지 보정
undistorted_left = cv2.undistort(img_left, mtx_left, dist_left)
undistorted_right = cv2.undistort(img_right, mtx_right, dist_right)

# 이미지 정합
img_left_rectified = cv2.remap(undistorted_left, map1_left, map2_left, cv2.INTER_LINEAR)
img_right_rectified = cv2.remap(undistorted_right, map1_right, map2_right, cv2.INTER_LINEAR)

cv2.imshow('Left Image (Rectified)',img_left_rectified)
cv2.imshow('Right Image (Rectified)', img_right_rectified)

stereo = cv2.StereoSGBM_create(minDisparity=0, numDisparities=16, blockSize=5)
disparity = stereo.compute(img_left_rectified, img_right_rectified)

disparity = cv2.normalize(disparity, disparity, 0, 255, cv2.NORM_MINMAX)
disparity = np.uint8(disparity)

disparity_colormap = cv2.applyColorMap(disparity, cv2.COLORMAP_JET)

cv2.imshow('Disparity Map (Colormap)', disparity_colormap)
cv2.imwrite('disparity_map_colormap.png', disparity_colormap)

cv2.waitKey(0)
cv2.destroyAllWindows()
'''
# 깊이 맵 계산 (disparity map)
# SGBM 사용
window_size = 5
min_disp = 16
num_disp = 112 - min_disp
block_size = 5

stereo = cv2.StereoSGBM_create(minDisparity=min_disp, numDisparities=num_disp, blockSize=block_size)
disparity = stereo.compute(img_left_rectified, img_right_rectified)

# 깊이 맵 시각화
disparity = (disparity - min_disp) / num_disp
cv2.imshow('Disparity Map', disparity)
cv2.waitKey(0)

# 보정된 이미지 시각화
cv2.imshow('Left Image (Rectified)', img_left_rectified)
cv2.imshow('Right Image (Rectified)', img_right_rectified)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
