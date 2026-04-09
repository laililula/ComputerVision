# SORT 알고리즘을 활용한 다중 객체 추적기 구현
# Dynamic Vision - Practice 15

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
import os

# ==================== SORT 추적기 클래스 ====================

class KalmanBoxTracker:
    """
    칼만 필터를 사용한 객체 추적 클래스입니다.
    bounding box의 위치와 속도를 예측합니다.
    """
    count = 0  # 추적 객체의 고유 ID 카운터
    
    def __init__(self, bbox):
        """
        bbox: [x1, y1, x2, y2] 형태의 경계 상자
        """
        self.bbox = bbox
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.frames_since_update = 0
        self.history = []
        
    def get_state(self):
        """현재 경계 상자 반환"""
        return self.bbox
    
    def update(self, new_bbox):
        """새로운 경계 상자로 위치 업데이트"""
        self.bbox = new_bbox
        self.frames_since_update = 0
        self.history.append(new_bbox)
    
    def predict(self):
        """다음 프레임의 위치 예측 (간단한 선형 외삽)"""
        # 이전 두 위치의 차이로부터 객체의 이동 벡터(속도)를 계산합니다
        # 이를 현재 위치에 더해 다음 프레임의 위치를 예측합니다 (칼만 필터 간소화 버전)
        if len(self.history) > 1:
            # 마지막 두 위치의 차이를 이용해 위치 변화량 계산
            dx = self.history[-1][0] - self.history[-2][0]
            dy = self.history[-1][1] - self.history[-2][1]
            dw = self.history[-1][2] - self.history[-2][2]
            dh = self.history[-1][3] - self.history[-2][3]
            # 현재 위치 + 이전 이동량 = 예측 위치
            predicted = [
                self.bbox[0] + dx,
                self.bbox[1] + dy,
                self.bbox[2] + dw,
                self.bbox[3] + dh
            ]
        else:
            # 히스토리가 없으면 현재 위치를 그대로 반환
            predicted = self.bbox
        
        # 이 추적 객체가 업데이트되지 않은 프레임 수를 증가
        self.frames_since_update += 1
        return predicted
    
    def increment_age(self):
        """업데이트되지 않은 프레임 카운트 증가"""
        self.frames_since_update += 1


class Sort:
    """
    SORT 추적기: Simple Online and Realtime Tracking
    칼만 필터와 헝가리안 알고리즘을 사용하여 객체를 추적합니다.
    """
    def __init__(self, max_age=30, min_hits=3):
        """
        max_age: 추적 객체의 최대 나이 (프레임 단위)
        min_hits: 객체를 인정하기 위한 최소 감지 횟수
        """
        self.trackers = []
        self.frame_count = 0
        self.max_age = max_age
        self.min_hits = min_hits
    
    def iou(self, bbox1, bbox2):
        """
        두 경계 상자 간의 IoU (Intersection over Union) 계산
        bbox: [x1, y1, x2, y2] 형태
        
        IoU는 두 객체가 얼마나 겹치는지 나타내는 지표입니다.
        이전 프레임의 객체와 현재 프레임의 검출 결과를 비교할 때 사용됩니다.
        """
        # 두 경계 상자의 교집합 영역의 좌상단 좌표 계산
        x1_inter = max(bbox1[0], bbox2[0])
        y1_inter = max(bbox1[1], bbox2[1])
        # 두 경계 상자의 교집합 영역의 우하단 좌표 계산
        x2_inter = min(bbox1[2], bbox2[2])
        y2_inter = min(bbox1[3], bbox2[3])
        
        # 교집합이 없으면 (겹치는 부분이 없으면) IoU = 0
        if x2_inter < x1_inter or y2_inter < y1_inter:
            return 0.0
        
        # 교집합의 면적 계산
        inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        
        # 각 경계 상자의 면적 계산
        bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        # 두 경계 상자의 합집합 면적 = 면적1 + 면적2 - 교집합
        union_area = bbox1_area + bbox2_area - inter_area
        
        if union_area == 0:
            return 0.0
        
        # IoU = 교집합 면적 / 합집합 면적
        return inter_area / union_area
    
    def update(self, detections):
        """
        새로운 프레임의 검출 결과로 추적 정보 업데이트
        detections: [[x1, y1, x2, y2, conf], ...] 형태의 배열
        
        SORT 알고리즘의 핵심 단계:
        1. 기존 추적 객체들의 위치를 예측
        2. 예측 위치와 현재 검출 결과의 유사성을 측정 (IoU 계산)
        3. 헝가리안 알고리즘으로 최적 매칭 수행
        4. 매칭된 객체는 업데이트, 미매칭 객체는 새 추적으로 생성
        """
        self.frame_count += 1
        
        if len(detections) == 0:
            # 검출된 객체가 없을 때는 모든 추적 객체의 나이 증가
            for tracker in self.trackers:
                tracker.increment_age()
            return []
        
        # 현재 추적 중인 모든 객체들의 다음 프레임 위치 예측
        # 각 추적 객체의 predict() 호출로 다음 위치 추정
        predicted_bboxes = [tracker.predict() for tracker in self.trackers]
        
        # IoU를 기반으로 비용 행렬 생성
        # 행: 기존 추적 객체, 열: 현재 검출 결과
        # 값: 1 - IoU (낮을수록 유사하므로 최소화 대상)
        iou_matrix = np.zeros((len(self.trackers), len(detections)))
        for i, pred_bbox in enumerate(predicted_bboxes):
            for j, det_bbox in enumerate(detections):
                # IoU가 높을수록 같은 객체일 가능성이 높으므로, 1 - IoU를 비용으로 사용
                iou_matrix[i, j] = 1 - self.iou(pred_bbox, det_bbox[:4])
        
        # 헝가리안 알고리즘으로 최적 매칭 찾기
        # 이 알고리즘은 전체 비용이 최소가 되도록 일대일 매칭을 수행합니다
        if iou_matrix.size > 0:
            t_indices, d_indices = linear_sum_assignment(iou_matrix)
        else:
            t_indices, d_indices = np.array([], dtype=int), np.array([], dtype=int)
        
        # 매칭된 추적 객체와 검출 결과 업데이트
        # 매칭되면 추적 객체의 위치를 새 검출 결과로 업데이트
        matched_detection_indices = set()
        for t_idx, d_idx in zip(t_indices, d_indices):
            # IoU 임계값 0.5 이상인 경우만 같은 객체로 판단
            if iou_matrix[t_idx, d_idx] < 0.5:
                self.trackers[t_idx].update(detections[d_idx][:4])
                matched_detection_indices.add(d_idx)
            else:
                # 일치도가 낮으면 업데이트하지 않고 나이만 증가
                self.trackers[t_idx].increment_age()
        
        # 매칭되지 않은 추적 객체들의 나이 증가
        # 이들이 max_age를 초과하면 나중에 제거됩니다
        for i, tracker in enumerate(self.trackers):
            if i not in t_indices:
                tracker.increment_age()
        
        # 매칭되지 않은 검출 결과를 새로운 추적 객체로 생성
        # 화면에 새로 나타난 객체는 새로운 추적 시작
        for j, detection in enumerate(detections):
            if j not in matched_detection_indices:
                new_tracker = KalmanBoxTracker(detection[:4])
                self.trackers.append(new_tracker)
        
        # 너무 오래된 추적 객체 제거
        # max_age 이상 업데이트되지 않은 객체는 화면을 벗어난 것으로 간주
        self.trackers = [t for t in self.trackers 
                        if t.frames_since_update <= self.max_age]
        
        # 결과 반환 (최소 감지 횟수를 만족하는 객체만)
        # 최근 업데이트된 객체(frames_since_update == 0)만 반환
        result = []
        for tracker in self.trackers:
            if tracker.frames_since_update == 0:
                bbox = tracker.get_state()
                result.append([bbox[0], bbox[1], bbox[2], bbox[3], tracker.id])
        
        return result


# ==================== YOLOv3 객체 검출기 ====================

class YOLOv3Detector:
    """YOLOv3를 사용한 객체 검출기"""
    def __init__(self, config_path, weights_path, conf_threshold=0.5, nms_threshold=0.4):
        """
        YOLOv3 모델 초기화
        config_path: yolov3.cfg 파일 경로
        weights_path: yolov3.weights 파일 경로
        """
        print("[*] YOLOv3 모델 로딩 중...")
        self.net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        print(f"[✓] YOLOv3 모델 로딩 완료")
    
    def detect(self, frame):
        """
        프레임에서 객체 검출
        반환: [[x1, y1, x2, y2, confidence], ...]
        
        YOLOv3 모델을 사용하여 프레임의 모든 객체를 검출합니다.
        """
        height, width = frame.shape[:2]
        
        # YOLOv3 입력을 위해 이미지를 blob으로 변환
        # YOLO는 정확한 416x416 크기의 입력이 필요합니다
        # 픽셀값을 0~1로 정규화하고, RGB 순서로 변환합니다
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), 
                                     swapRB=True, crop=False)
        self.net.setInput(blob)
        
        # 신경망의 모든 출력층 이름 가져오기
        output_layers = self.net.getUnconnectedOutLayersNames()
        
        # 신경망에서 예측 수행
        outs = self.net.forward(output_layers)
        
        # 검출된 객체들의 경계 상자와 신뢰도 저장
        boxes = []
        confidences = []
        
        # 모든 출력층의 결과 처리
        for out in outs:
            for detection in out:
                # 각 detection은 [x, y, w, h, confidence, class_scores...]
                # detection[5:]는 각 클래스의 신뢰도 점수입니다
                scores = detection[5:]
                # 가장 높은 신뢰도를 가진 클래스 선택
                class_id = np.argmax(scores)
                # 선택된 클래스의 신뢰도
                confidence = scores[class_id]
                
                # 신뢰도 임계값을 넘은 것만 처리
                if confidence > self.conf_threshold:
                    # 중심 좌표(정규화 좌표)를 픽셀 좌표로 변환
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    # 중심 좌표를 좌상단-우하단 좌표로 변환
                    x1 = max(0, center_x - w // 2)
                    y1 = max(0, center_y - h // 2)
                    x2 = min(width, center_x + w // 2)
                    y2 = min(height, center_y + h // 2)
                    
                    boxes.append([x1, y1, x2, y2])
                    confidences.append(float(confidence))
        
        # NMS (Non-Maximum Suppression): 중복된 검출 제거
        # 겹치는 여러 상자 중 가장 신뢰도가 높은 것만 유지
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 
                                   self.conf_threshold, self.nms_threshold)
        
        # NMS 후 남은 검출 결과만 반환
        detections = []
        if len(indices) > 0:
            for i in indices.flatten():
                detections.append(boxes[i] + [confidences[i]])
        
        return detections


# ==================== 메인 함수 ====================

def main():
    """메인 함수: SORT를 이용한 다중 객체 추적"""
    
    print("\n" + "="*60)
    print("SORT 알고리즘을 활용한 다중 객체 추적기")
    print("="*60 + "\n")
    
    # YOLOv3 모델 파일 경로 설정
    config_path = r"C:\Users\alsrb\ComputerVision\L06\yolov3.cfg"
    weights_path = r"C:\Users\alsrb\ComputerVision\L06\yolov3.weights"
    
    # 파일 존재 확인
    if not os.path.exists(config_path):
        print(f"[✗] 설정 파일을 찾을 수 없습니다: {config_path}")
        return
    if not os.path.exists(weights_path):
        print(f"[✗] 가중치 파일을 찾을 수 없습니다: {weights_path}")
        return
    
    # YOLOv3 검출기 초기화
    # 신경망 구조와 사전 훈련된 가중치 로드
    try:
        detector = YOLOv3Detector(config_path, weights_path)
    except Exception as e:
        print(f"[✗] YOLOv3 초기화 실패: {e}")
        return
    
    # SORT 추적기 초기화
    # max_age: 추적 유지 시간(프레임), min_hits: 객체 인정 기준
    sort_tracker = Sort(max_age=30, min_hits=3)
    
    # 웹캠 연결
    print("[*] 웹캠 연결 중...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("[✗] 웹캠을 열 수 없습니다")
        return
    
    print("[✓] 웹캠 연결 완료\n")
    print("[ 종료: ESC 키 또는 Q 키 ]")
    print("="*60 + "\n")
    
    # 추적 객체마다 다른 색상 할당을 위한 딕셔너리
    colors = {}
    
    # 비디오 처리 메인 루프
    frame_count = 0
    while True:
        # 웹캠에서 프레임 읽기
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        height, width = frame.shape[:2]
        
        # 처리 속도 향상을 위해 프레임 크기 조정
        # 원본 크기를 유지하기 위해 비율을 계산합니다
        resized_frame = cv2.resize(frame, (640, 480))
        scale_x = width / 640
        scale_y = height / 480
        
        # YOLOv3로 객체 검출
        # 리사이즈된 프레임에서 검출하므로 좌표 변환 필요
        detections = detector.detect(resized_frame)
        
        # 검출 좌표를 원본 프레임 크기로 변환
        scaled_detections = []
        for det in detections:
            scaled_det = [
                det[0] * scale_x,
                det[1] * scale_y,
                det[2] * scale_x,
                det[3] * scale_y,
                det[4]
            ]
            scaled_detections.append(scaled_det)
        
        # SORT로 추적 정보 업데이트
        # 현재 프레임의 검출 결과를 기반으로 추적 객체 매칭
        tracked_objects = sort_tracker.update(scaled_detections)
        
        # 추적된 객체들을 원본 프레임에 시각화
        for obj in tracked_objects:
            x1, y1, x2, y2, obj_id = map(int, obj)
            
            # 객체마다 고유한 색상 할당
            # 같은 ID는 같은 색상으로 표시되어 추적이 명확합니다
            if obj_id not in colors:
                colors[obj_id] = tuple(np.random.randint(0, 255, 3).tolist())
            
            color = colors[obj_id]
            
            # 경계 상자 그리기
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # 객체 ID 표시
            label = f"ID: {obj_id}"
            cv2.putText(frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # 프레임 정보 표시 (현재 프레임 번호, 추적 객체 수)
        info_text = f"Frame: {frame_count} | Tracked Objects: {len(tracked_objects)}"
        cv2.putText(frame, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 처리된 프레임을 화면에 표시
        cv2.imshow('Multi-Object Tracking with SORT', frame)
        
        # 키 입력 처리
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):  # ESC 또는 Q 키
            break
    
    # 리소스 해제
    cap.release()
    cv2.destroyAllWindows()
    print("\n[✓] 프로그램 종료")


if __name__ == "__main__":
    main()
