import cv2, dlib, sys
import numpy as np

scaler = 0.2 # img 크기 조절용 변수 

# initialize face detector and shape predictor
detector = dlib.get_frontal_face_detector() # 얼굴 디텍터 모듈 초기화
predictor = dlib.shape_predictor('E:\\face_detector-master\\shape_predictor_68_face_landmarks.dat')
# 얼굴 특징점 모듈을 초기화

# load video
cap = cv2.VideoCapture('E:\\face_detector-master\\samples\\girl.mp4')
ret, img2 = cap.read()
# print(img2.shape) -> 이미지 크기(가로,세로,RGB)를 확인


# Save video (저장할 비디오 파일 타입과 변수를 지정)
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
out = cv2.VideoWriter('E:\\face_detector-master\\results\\output3.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS), (img2.shape[1], img2.shape[0]))

# Save facial landmarks video ( facial landmarks 비디오 파일 타입과 변수를 지정)
fourcc2 = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
out2 = cv2.VideoWriter('E:\\face_detector-master\\faciallandmarks\\output_faciallandmarks3.mp4', fourcc2, cap.get(cv2.CAP_PROP_FPS), (img2.shape[1], img2.shape[0]))

# load overlay image
overlay = cv2.imread('E:\\face_detector-master\\samples\\teemo.png', cv2.IMREAD_UNCHANGED)
 
# overlay function
def overlay_transparent(background_img, img_to_overlay_t, x, y, overlay_size=None):
  try:    
      
    bg_img = background_img.copy()
    # convert 3 channels to 4 channels
    if bg_img.shape[2] == 3:
      bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2BGRA)
  
    if overlay_size is not None:
      img_to_overlay_t = cv2.resize(img_to_overlay_t.copy(), overlay_size)
  
    b, g, r, a = cv2.split(img_to_overlay_t)
  
    mask = cv2.medianBlur(a, 5)
  
    h, w, _ = img_to_overlay_t.shape
    roi = bg_img[int(y-h/2):int(y+h/2), int(x-w/2):int(x+w/2)]
  
    img1_bg = cv2.bitwise_and(roi.copy(), roi.copy(), mask=cv2.bitwise_not(mask))
    img2_fg = cv2.bitwise_and(img_to_overlay_t, img_to_overlay_t, mask=mask)
  
    bg_img[int(y-h/2):int(y+h/2), int(x-w/2):int(x+w/2)] = cv2.add(img1_bg, img2_fg)
  
    # convert 4 channels to 4 channels
    bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGRA2BGR)
  
    return bg_img

  except Exception : return background_img
  
face_roi = []
face_sizes = []
# face_roi -> 얼굴 부분만 자르는 부분

# loop
while True:
  # read frame buffer from video
  ret, img = cap.read()
  # ret : frame capture 결과(boolean)
  # 동영상을 제대로 읽어오면 ret가 1(True), 제대로 못읽어오면 ret가 0(False)
  
  if not ret:
    break

  # resize frame
  img = cv2.resize(img, (int(img.shape[1] * scaler), int(img.shape[0] * scaler)))
  
  # cv2.resize(원본 이미지, 결과 이미지 크기, 보간법)
  # int(img.shape[1] * scaler) -> 가로(width)
  # int(img.shape[0] * scaler) -> 세로(height)
  
  ori = img.copy() # img를 ori라는 변수에 복사 
  
  # find faces 
  if len(face_roi) == 0: # 처음에는 face_roi = [] 이므로, 이 부분이 먼저 실행된다.
    faces = detector(img, 1)
  else:
    roi_img = img[face_roi[0]:face_roi[1], face_roi[2]:face_roi[3]]
    
# =============================================================================
#     print('face_roi 좌표값들')
#     print(face_roi[0]) # int(min_coords[1] - face_size / 2) # 세로(행)
#     print(face_roi[1]) # int(max_coords[1] + face_size / 2) # 세로(행)
#     print(face_roi[2]) # int(min_coords[0] - face_size / 2) # 가로(열)
#     print(face_roi[3]) # int(max_coords[0] + face_size / 2) # 가로(열)
#     print('\n')
#     
#     print('roi_img = \n', roi_img)
#     print('\n')
# =============================================================================
    
    #cv2.imshow('roi', roi_img)
    faces = detector(roi_img)

  # no faces
  if len(faces) == 0:
    print('no faces!')

  # find facial landmarks
  for face in faces:
    if len(face_roi) == 0:
      dlib_shape = predictor(img, face) # img의 face 영역안의 얼굴 특징점을 찾기
      shape_2d = np.array([[p.x, p.y] for p in dlib_shape.parts()])
      # dlib 객체를 연산이 쉽도록 numpy 객체로 변환
      
    else:
      dlib_shape = predictor(roi_img, face)
      shape_2d = np.array([[p.x + face_roi[2], p.y + face_roi[0]] for p in dlib_shape.parts()])
      
    for s in shape_2d:
      cv2.circle(img, center=tuple(s), radius=1, color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
      # 64개의 얼굴의 특징점들을 나타내주는 코드
      
    # compute face center
    center_x, center_y = np.mean(shape_2d, axis=0).astype(np.int) 

    # compute face boundaries
    min_coords = np.min(shape_2d, axis=0)  # 좌상단 좌표
    max_coords = np.max(shape_2d, axis=0)  # 우하단 좌표
    
    # print('좌상단좌표 =', min_coords) # (234,68) (가로좌표, 세로좌표)
    # print('우하단좌표 =', max_coords) # (318,156) (가로좌표, 세로좌표)
    
    # draw min, max coords
    cv2.circle(img, center=tuple(min_coords), radius=1, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
    cv2.circle(img, center=tuple(max_coords), radius=1, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
    
    # draw center coords
    cv2.circle(img, center=tuple((center_x, center_y)), radius=1, color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
     
    # compute face size
    face_size = max(max_coords - min_coords)
    # print('얼굴크기 = ' , face_size, end = '\n\n')
    face_sizes.append(face_size)
    
    if len(face_sizes) > 10:
      del face_sizes[0] # face_sizes의 length(len)가 10이 넘어가면 맨 앞의 요소를 삭제시킴.
      
    mean_face_size = int(np.mean(face_sizes) * 1.8) 
    
    # 위의 이부분에서 라이언 크기를 조절할 수 있음.
    
    # print('평균 얼굴 사이즈 = ', mean_face_size)
    
    # 평균 얼굴 사이즈를 계산(라이언 씌우기 위해서 구함) & 그리고 임의로 1.8을 곱함(라이언 크기 키우려고)

    # compute face roi
    face_roi = np.array([int(min_coords[1] - face_size / 2), int(max_coords[1] + face_size / 2), \
                         int(min_coords[0] - face_size / 2), int(max_coords[0] + face_size / 2)])
    face_roi = np.clip(face_roi, 0, 10000)

    # draw overlay on face (라이언 씌우는 부분)
    result = overlay_transparent(ori, overlay, center_x + 8, center_y - 15, \
                                 overlay_size=(mean_face_size, mean_face_size))
        
    # center_x + 8, center_y - 25 부분으로 위치를 조정
    # overlay_size=(mean_face_size, mean_face_size) 부분으로 라이언 가면 부분 크기 조정
    
  # visualize(결과 보여주기)
  cv2.imshow('original', ori)
  cv2.imshow('facial landmarks', img)
  cv2.imshow('result', result)

  # result를 앞에 있는 out 변수와 framesize를 꼭 같게 맞추어 주어야 된다.
  # framesize가 같지 않으면 저장이 안된다. (0 or 1byte)
  result = cv2.resize(result, (img2.shape[1], img2.shape[0]))    
  img = cv2.resize(img, (img2.shape[1], img2.shape[0]))
  
  # out변수에 result(라이언가면 씌운 영상)을 기록한다.
  out.write(result)
  out2.write(img)
  
  if cv2.waitKey(1) == ord('q'):
   #sys.exit(1)
    break

print("[INFO] cleaning up...")

cap.release() # 사용한 자원을 해제
out.release() # 사용한 자원을 해제
cv2.destroyAllWindows() # 모든 창을 닫는다

print("[INFO] Save Complete & The END")