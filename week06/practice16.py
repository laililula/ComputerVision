# Mediapipe를 활용한 얼굴 랜드마크 추출 및 시각화
# Dynamic Vision - Practice 16

# OpenCV: 이미지/비디오 처리
import cv2
# Mediapipe: Google의 얼굴 인식 프레임워크
import mediapipe as mp

# ==================== Mediapipe 초기화 ====================

# Mediapipe의 FaceMesh 모듈을 설정합니다
# FaceMesh: 얼굴의 468개 3D 랜드마크를 검출합니다
mp_face_mesh = mp.solutions.face_mesh
# 검출된 결과를 시각화하기 위한 유틸리티
mp_drawing = mp.solutions.drawing_utils
# 그리기 스타일 설정
mp_drawing_styles = mp.solutions.drawing_styles


# ==================== 메인 함수 ====================

def main():
    """메인 함수: Mediapipe를 이용한 얼굴 랜드마크 추출"""
    
    print("\n" + "="*60)
    print("Mediapipe를 활용한 얼굴 랜드마크 추출 및 시각화")
    print("="*60 + "\n")
    
    print("[*] 웹캠 연결 중...")
    
    # OpenCV를 사용하여 웹캠으로부터 실시간 영상을 캡처합니다
    # VideoCapture(0): 기본 웹캠 (0번 카메라)
    cap = cv2.VideoCapture(0)
    
    # 웹캠 연결 확인
    if not cap.isOpened():
        print("[✗] 웹캠을 열 수 없습니다")
        return
    
    print("[✓] 웹캠 연결 완료\n")
    print("[ 종료: ESC 키 ]")
    print("="*60 + "\n")
    
    # Mediapipe의 FaceMesh 모듈을 사용하여 얼굴 랜드마크 검출기를 초기화합니다
    print("[*] Mediapipe FaceMesh 모델 로딩 중...")
    with mp_face_mesh.FaceMesh(
        # static_image_mode=False: 비디오 입력 모드 (프레임마다 처리)
        static_image_mode=False,
        # max_num_faces=2: 최대 2개의 얼굴 동시 검출
        max_num_faces=2,
        # refine_landmarks=True: 상세한 랜드마크 포함 (홍채, 눈썹 등)
        refine_landmarks=True,
        # 얼굴 검출 신뢰도 임계값 (50% 이상만 인식)
        min_detection_confidence=0.5,
        # 객체 추적 신뢰도 임계값
        min_tracking_confidence=0.5
    ) as face_mesh:
        
        print("[✓] 모델 로딩 완료\n")
        
        # 프레임 카운트
        frame_count = 0
        
        # 실시간 비디오 처리 메인 루프
        while True:
            # 웹캠에서 한 프레임 읽기
            ret, frame = cap.read()
            
            # 프레임 읽기 실패 확인
            if not ret:
                print("[✗] 프레임을 읽을 수 없습니다")
                break
            
            frame_count += 1
            # 프레임의 높이, 너비, 채널 수 획득
            height, width, _ = frame.shape
            
            # BGR을 RGB로 변환 (Mediapipe는 RGB 입력을 요구)
            # OpenCV는 BGR 순서로 이미지를 저장하므로 변환 필요
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 얼굴 랜드마크를 검출합니다
            # process(): Mediapipe의 핵심 함수로 실제 얼굴 인식 수행
            results = face_mesh.process(frame_rgb)
            
            # 검출된 얼굴 랜드마크를 실시간 영상에 점으로 표시합니다
            # 검출된 얼굴의 개수
            num_faces = 0
            
            # 얼굴이 검출되었는지 확인
            if results.multi_face_landmarks:
                # 검출된 각 얼굴에 대해 처리
                for face_landmarks in results.multi_face_landmarks:
                    # 검출된 얼굴 수 카운트
                    num_faces += 1
                    
                    # 메시와 연결선을 그립니다
                    # 468개의 얼굴 포인트가 삼각형으로 연결된 메시 표시
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        # FACEMESH_TESSELATION: 얼굴 표면의 삼각형 메시
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,  # 랜드마크 점은 그리지 않음
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                    )
                    
                    # 각 랜드마크를 점으로 시각화합니다
                    # 468개의 개별 포인트를 노란색 점으로 표시
                    for landmark in face_landmarks.landmark:
                        # 정규화된 좌표(0~1)를 픽셀 좌표로 변환
                        x = int(landmark.x * width)
                        y = int(landmark.y * height)
                        # 반지름 1의 원(점)으로 표시, 채워진 원(-1)
                        # BGR 순서: (255, 255, 0) = 노란색
                        cv2.circle(frame, (x, y), 1, (255, 255, 0), -1)
            
            # 프레임 정보 표시
            # 현재 처리 중인 프레임 번호와 검출된 얼굴 수 표시
            info_text = f"Frame: {frame_count} | Detected Faces: {num_faces}"
            cv2.putText(frame, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 처리된 프레임을 화면에 출력
            cv2.imshow('Face Landmarks Detection with Mediapipe', frame)
            
            # ESC 키를 누르면 프로그램이 종료되도록 설정합니다
            # cv2.waitKey(1): 1ms 동안 키 입력 대기
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC 키의 ASCII 코드 = 27
                break
    
    # 리소스 해제
    # 웹캠 연결 종료
    cap.release()
    # 모든 OpenCV 창 닫기
    cv2.destroyAllWindows()
    print("\n[✓] 프로그램 종료")


if __name__ == "__main__":
    main()
