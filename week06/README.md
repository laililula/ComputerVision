# Week06 Practice README

이 README 파일은 week06 폴더의 practice15.py, practice16.py 파일에 대한 내용을 정리한 문서입니다.  
각 실습마다 전체 코드, 주요 코드 설명, 실행 결과 영역을 포함합니다.

---

# Practice 15: SORT 알고리즘을 활용한 다중 객체 추적기 구현

## 전체 코드

```python
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
```

## 주요 코드 설명

### 1. `KalmanBoxTracker` 클래스 - 개별 객체의 상태를 관리하는 추적 단위

```python
class KalmanBoxTracker:
    count = 0
    
    def __init__(self, bbox):
        self.bbox = bbox
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.frames_since_update = 0
        self.history = []
```

이 클래스는 **화면에 등장한 객체 1개를 담당하는 추적 단위**입니다.
즉, 사람 1명, 자동차 1대처럼 검출된 객체마다 `KalmanBoxTracker` 인스턴스가 하나씩 만들어집니다.

- `self.bbox`:
  - 현재 객체의 바운딩 박스 좌표 `[x1, y1, x2, y2]`를 저장합니다.
  - 추적 결과를 그릴 때 가장 직접적으로 사용되는 상태값입니다.

- `self.id`:
  - 각 객체에 고유 번호를 부여합니다.
  - 같은 객체가 다음 프레임에서도 같은 ID를 유지하도록 하기 위한 핵심 값입니다.

- `KalmanBoxTracker.count`:
  - 새 객체가 생성될 때마다 1씩 증가하는 전역 카운터 역할을 합니다.
  - 예를 들어 첫 번째 객체는 ID 0, 두 번째는 ID 1처럼 할당됩니다.

- `self.frames_since_update`:
  - 최근 몇 프레임 동안 이 객체가 새 검출 결과로 갱신되지 않았는지 기록합니다.
  - 이 값이 커지면 화면에서 사라졌다고 보고 제거 후보가 됩니다.

- `self.history`:
  - 이전 프레임들의 위치를 저장합니다.
  - 이 코드에서는 복잡한 칼만 필터 대신, 이전 위치 변화량을 이용한 **간단한 선형 외삽**에 활용됩니다.

즉, 이 클래스는 “객체 하나의 현재 위치, 과거 이동 이력, 고유 ID, 최근 갱신 여부”를 묶어서 관리하는 구조라고 보면 됩니다.

### 2. `predict()` - 이전 이동량을 이용해 다음 위치를 예측

```python
def predict(self):
    if len(self.history) > 1:
        dx = self.history[-1][0] - self.history[-2][0]
        dy = self.history[-1][1] - self.history[-2][1]
        dw = self.history[-1][2] - self.history[-2][2]
        dh = self.history[-1][3] - self.history[-2][3]
        predicted = [
            self.bbox[0] + dx,
            self.bbox[1] + dy,
            self.bbox[2] + dw,
            self.bbox[3] + dh
        ]
    else:
        predicted = self.bbox

    self.frames_since_update += 1
    return predicted
```

이 함수는 **다음 프레임에서 객체가 어디에 있을지 미리 추정**합니다.

이 코드의 핵심 아이디어는 다음과 같습니다.

- 최근 두 프레임의 위치 차이를 이용해 이동량을 계산합니다.
- 그 이동량을 현재 위치에 더하면 다음 프레임의 예상 위치를 얻을 수 있습니다.

예를 들어,
- 이전 프레임의 좌표가 `[100, 50, 180, 150]`
- 현재 프레임의 좌표가 `[110, 55, 190, 155]`
이라면,
- 이동량은 `(+10, +5, +10, +5)`가 됩니다.
- 따라서 다음 프레임 예측은 `[120, 60, 200, 160]`처럼 계산됩니다.

이것은 엄밀한 칼만 필터 전체 구현은 아니지만, **“직전까지의 움직임이 다음 프레임에도 비슷하게 유지된다”**는 가정을 사용한 매우 단순하고 직관적인 예측 방식입니다.

또한 `self.frames_since_update += 1`이 여기서 증가하는 이유는,
- 예측은 했지만 아직 실제 검출 결과로 갱신된 것은 아니기 때문입니다.
- 이후 `update()`가 호출되면 다시 0으로 초기화됩니다.

### 3. `iou()` - 두 바운딩 박스가 얼마나 겹치는지 계산

```python
def iou(self, bbox1, bbox2):
    x1_inter = max(bbox1[0], bbox2[0])
    y1_inter = max(bbox1[1], bbox2[1])
    x2_inter = min(bbox1[2], bbox2[2])
    y2_inter = min(bbox1[3], bbox2[3])

    if x2_inter < x1_inter or y2_inter < y1_inter:
        return 0.0

    inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union_area = bbox1_area + bbox2_area - inter_area

    return inter_area / union_area
```

IoU는 **Intersection over Union**의 약자로,
두 바운딩 박스가 얼마나 많이 겹치는지 수치로 나타내는 대표적인 지표입니다.

공식은 다음과 같습니다.

$$
IoU = rac{	ext{교집합 면적}}{	ext{합집합 면적}}
$$

해석은 다음과 같습니다.

- `IoU = 1`에 가까움: 거의 같은 위치의 박스
- `IoU = 0`에 가까움: 거의 겹치지 않음
- 값이 클수록 같은 객체일 가능성이 높음

코드 흐름을 보면,

1. `max()`와 `min()`으로 두 박스의 겹치는 부분 좌표를 구합니다.
2. 겹치는 영역이 없으면 바로 `0.0`을 반환합니다.
3. 교집합 면적과 각 박스의 면적을 구합니다.
4. 합집합 면적을 계산한 뒤 `inter_area / union_area`를 반환합니다.

이 함수는 SORT에서 매우 중요합니다.
왜냐하면 “이전 프레임의 추적 결과”와 “현재 프레임의 검출 결과”를 연결할 때,
서로 얼마나 겹치는지를 기준으로 같은 객체인지 판단하기 때문입니다.

### 4. `Sort.update()` - SORT 알고리즘 전체 흐름을 담당하는 핵심 함수

```python
def update(self, detections):
    self.frame_count += 1

    if len(detections) == 0:
        for tracker in self.trackers:
            tracker.increment_age()
        return []

    predicted_bboxes = [tracker.predict() for tracker in self.trackers]
    iou_matrix = np.zeros((len(self.trackers), len(detections)))
```

이 함수는 **현재 프레임의 검출 결과를 받아서 기존 추적기들과 연결하고, 새 추적기를 만들고, 오래된 추적기를 제거하는 전체 과정**을 수행합니다.

SORT의 핵심 절차를 단계별로 정리하면 다음과 같습니다.

#### (1) 기존 추적 객체들의 다음 위치 예측

```python
predicted_bboxes = [tracker.predict() for tracker in self.trackers]
```

- 현재까지 관리 중인 모든 tracker에 대해 `predict()`를 호출합니다.
- 이렇게 얻은 예측 위치는 “이전 프레임의 객체가 이번 프레임에서는 여기쯤 있을 것”이라는 가설입니다.

#### (2) 비용 행렬(cost matrix) 생성

```python
iou_matrix[i, j] = 1 - self.iou(pred_bbox, det_bbox[:4])
```

- 행: 기존 tracker
- 열: 현재 detections
- 값: `1 - IoU`

왜 `1 - IoU`를 쓰는가?
헝가리안 알고리즘은 **비용이 작을수록 좋은 매칭**으로 보기 때문입니다.

- IoU가 크다 → 많이 겹친다 → 같은 객체일 가능성 높다
- 따라서 비용은 작아야 한다
- 그래서 `cost = 1 - IoU`로 바꿉니다.

예를 들어,
- IoU가 0.9면 비용은 0.1
- IoU가 0.1이면 비용은 0.9
가 되어, 더 잘 겹치는 조합이 더 유리해집니다.

#### (3) 헝가리안 알고리즘으로 최적 매칭 수행

```python
t_indices, d_indices = linear_sum_assignment(iou_matrix)
```

`linear_sum_assignment()`는 SciPy에서 제공하는 헝가리안 알고리즘 구현입니다.
이 함수는 전체 비용이 최소가 되도록 tracker와 detection을 **일대일 매칭**합니다.

이 단계가 필요한 이유는,
단순히 각 tracker가 가장 가까운 detection을 독립적으로 고르면
- 두 tracker가 같은 detection에 붙으려 하거나,
- 전체적으로 비효율적인 짝짓기가 나올 수 있기 때문입니다.

헝가리안 알고리즘은 전체 조합을 고려하여 가장 합리적인 매칭을 찾아줍니다.

#### (4) 매칭 결과 반영

```python
for t_idx, d_idx in zip(t_indices, d_indices):
    if iou_matrix[t_idx, d_idx] < 0.5:
        self.trackers[t_idx].update(detections[d_idx][:4])
        matched_detection_indices.add(d_idx)
    else:
        self.trackers[t_idx].increment_age()
```

여기서 `iou_matrix[t_idx, d_idx] < 0.5`라는 조건은,
비용이 0.5보다 작다는 뜻이므로 결국

$$
1 - IoU < 0.5 \Rightarrow IoU > 0.5
$$

를 의미합니다.

즉,
- IoU가 0.5보다 크면 같은 객체라고 보고 업데이트
- 그렇지 않으면 억지 매칭으로 간주하고 나이만 증가시킵니다.

이 과정 덕분에 잘못된 ID 연결을 어느 정도 방지할 수 있습니다.

#### (5) 매칭되지 않은 tracker 처리

```python
for i, tracker in enumerate(self.trackers):
    if i not in t_indices:
        tracker.increment_age()
```

기존 tracker 중 이번 프레임에서 어떤 detection과도 연결되지 못한 경우입니다.
보통 이런 상황은 다음을 의미합니다.

- 객체가 잠시 가려짐
- 검출기가 해당 객체를 놓침
- 객체가 화면 밖으로 나감

이 tracker를 바로 삭제하지 않고 나이만 증가시키는 이유는,
짧은 순간의 검출 실패에도 ID가 끊기지 않도록 하기 위해서입니다.

#### (6) 매칭되지 않은 detection은 새 tracker 생성

```python
for j, detection in enumerate(detections):
    if j not in matched_detection_indices:
        new_tracker = KalmanBoxTracker(detection[:4])
        self.trackers.append(new_tracker)
```

현재 프레임에서 새롭게 등장한 객체는 기존 tracker와 연결되지 않습니다.
이 경우 새 `KalmanBoxTracker`를 만들어서 새로운 ID를 부여합니다.

즉, 화면에 처음 나타난 사람이나 물체는 이 단계에서 추적 시작됩니다.

#### (7) 오래된 tracker 제거

```python
self.trackers = [t for t in self.trackers if t.frames_since_update <= self.max_age]
```

오랫동안 갱신되지 않은 tracker는 제거합니다.

- `max_age=30`이면,
  30프레임 이상 검출되지 않은 객체는 더 이상 존재하지 않는다고 가정합니다.
- 이 값이 너무 작으면 잠깐 가려진 객체 ID가 쉽게 끊기고,
- 너무 크면 이미 사라진 객체가 오래 남아 있을 수 있습니다.

#### (8) 최종 반환값 구성

```python
for tracker in self.trackers:
    if tracker.frames_since_update == 0:
        bbox = tracker.get_state()
        result.append([bbox[0], bbox[1], bbox[2], bbox[3], tracker.id])
```

최종적으로는 **이번 프레임에서 실제로 업데이트된 tracker만 반환**합니다.
즉, 예측만 된 객체가 아니라 현재 검출 결과와 성공적으로 연결된 객체만 화면에 표시합니다.

### 5. `YOLOv3Detector.__init__()` - 사전학습된 검출 모델 로드

```python
self.net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
```

이 부분은 OpenCV DNN 모듈을 이용하여 **YOLOv3 객체 검출 모델을 불러오는 초기화 단계**입니다.

- `config_path`:
  - YOLOv3 네트워크 구조가 정의된 `.cfg` 파일 경로입니다.
- `weights_path`:
  - 미리 학습된 가중치가 저장된 `.weights` 파일 경로입니다.
- `readNetFromDarknet()`:
  - Darknet 형식의 YOLO 모델을 OpenCV에서 사용할 수 있는 네트워크 객체로 만듭니다.
- `DNN_BACKEND_OPENCV`:
  - OpenCV 자체 backend를 사용합니다.
- `DNN_TARGET_CPU`:
  - CPU에서 추론을 수행하도록 지정합니다.

또한,

```python
self.conf_threshold = conf_threshold
self.nms_threshold = nms_threshold
```

는 검출 후 필터링 기준을 저장합니다.

- `conf_threshold=0.5`:
  - 신뢰도가 50% 이상인 검출만 유지합니다.
- `nms_threshold=0.4`:
  - NMS 수행 시 겹침 허용 기준입니다.

### 6. `detect()` - YOLOv3를 이용해 현재 프레임에서 객체를 검출

```python
blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
self.net.setInput(blob)
output_layers = self.net.getUnconnectedOutLayersNames()
outs = self.net.forward(output_layers)
```

이 함수는 입력 프레임을 받아 **객체 검출 결과를 `[x1, y1, x2, y2, confidence]` 형태로 반환**합니다.

처리 순서는 다음과 같습니다.

#### (1) 이미지 → blob 변환

`blobFromImage()`는 OpenCV DNN이 사용할 수 있는 입력 텐서 형태로 이미지를 변환합니다.

- `1/255.0`: 픽셀값을 0~1 범위로 정규화
- `(416, 416)`: YOLOv3 입력 크기에 맞게 리사이즈
- `swapRB=True`: OpenCV의 BGR을 RGB 순서로 변경
- `crop=False`: 비율을 억지로 자르지 않고 전체 이미지를 유지

즉, 원본 프레임을 신경망 입력 형식에 맞춰 전처리하는 단계입니다.

#### (2) 출력층 가져오기 및 추론

```python
output_layers = self.net.getUnconnectedOutLayersNames()
outs = self.net.forward(output_layers)
```

YOLO는 여러 출력층에서 결과를 내므로,
그 출력층 이름을 가져와서 한 번에 forward합니다.

`outs` 안에는 여러 scale에서 검출된 후보들이 들어 있습니다.
각 detection은 대체로 다음 구조를 가집니다.

```python
[x, y, w, h, objectness, class_score1, class_score2, ...]
```

#### (3) 클래스 점수 중 최고값 선택

```python
scores = detection[5:]
class_id = np.argmax(scores)
confidence = scores[class_id]
```

- `scores`는 각 클래스에 대한 점수입니다.
- `np.argmax(scores)`는 가장 높은 점수를 가진 클래스 인덱스를 찾습니다.
- 그 클래스의 점수를 `confidence`로 사용합니다.

이 코드에서는 클래스 이름을 따로 출력하지 않고,
“객체가 있다”는 사실과 바운딩 박스만 사용하여 추적합니다.

#### (4) 중심 좌표 → 모서리 좌표 변환

```python
center_x = int(detection[0] * width)
center_y = int(detection[1] * height)
w = int(detection[2] * width)
h = int(detection[3] * height)

x1 = max(0, center_x - w // 2)
y1 = max(0, center_y - h // 2)
x2 = min(width, center_x + w // 2)
y2 = min(height, center_y + h // 2)
```

YOLO의 출력은 보통 **중심점 좌표 + 너비/높이** 형식입니다.
하지만 이후 IoU 계산이나 사각형 그리기는 `[x1, y1, x2, y2]` 형식이 더 편하므로 변환합니다.

- `(center_x, center_y)`: 박스 중심
- `(w, h)`: 박스 크기
- `x1, y1`: 좌상단
- `x2, y2`: 우하단

또한 `max(0, ...)`, `min(width, ...)`를 사용해 이미지 밖으로 좌표가 벗어나지 않도록 보정합니다.

#### (5) NMS로 중복 박스 제거

```python
indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.nms_threshold)
```

객체 하나에 대해 비슷한 박스가 여러 개 나오는 경우가 많기 때문에,
겹치는 후보 중 **가장 신뢰도가 높은 박스만 남기는 과정**이 필요합니다.
이것이 NMS(Non-Maximum Suppression)입니다.

정리하면 `detect()`는
1. 입력 전처리
2. YOLO 추론
3. 후보 박스 추출
4. 좌표 변환
5. NMS 적용
을 거쳐 최종 검출 결과를 만듭니다.

### 7. `main()` - 검출 결과와 추적 결과를 연결하는 전체 실행 흐름

```python
cap = cv2.VideoCapture(0)
detector = YOLOv3Detector(config_path, weights_path)
sort_tracker = Sort(max_age=30, min_hits=3)
```

`main()`은 프로그램 전체를 실행하는 함수로,
웹캠 입력 → 객체 검출 → 좌표 보정 → 추적 → 시각화 → 종료 처리의 흐름을 담당합니다.

핵심 순서는 다음과 같습니다.

#### (1) 웹캠 연결 및 모델 초기화

- `cv2.VideoCapture(0)`으로 기본 카메라를 엽니다.
- YOLOv3Detector로 검출 모델을 준비합니다.
- `Sort()`로 다중 객체 추적기를 생성합니다.

#### (2) 프레임 리사이즈 및 좌표 비율 계산

```python
resized_frame = cv2.resize(frame, (640, 480))
scale_x = width / 640
scale_y = height / 480
```

검출 속도를 높이기 위해 프레임을 640×480으로 줄인 뒤 YOLO를 적용합니다.
그 대신, 검출된 좌표는 축소된 프레임 기준이므로 원본 프레임에 그리기 전에 다시 되돌려야 합니다.

- `scale_x`, `scale_y`는 축소 좌표를 원본 좌표로 환산하기 위한 비율입니다.

#### (3) 검출 좌표를 원본 크기로 복원

```python
scaled_det = [
    det[0] * scale_x,
    det[1] * scale_y,
    det[2] * scale_x,
    det[3] * scale_y,
    det[4]
]
```

이 단계가 필요한 이유는,
검출은 축소 프레임에서 했지만 화면 표시는 원본 프레임에 하기 때문입니다.
따라서 박스 위치를 원래 크기로 다시 변환해야 정확히 맞습니다.

#### (4) 추적기 업데이트

```python
tracked_objects = sort_tracker.update(scaled_detections)
```

현재 프레임의 검출 결과를 SORT에 넣으면,
- 기존 ID 유지 여부 판단
- 새 객체 생성
- 사라진 객체 정리
가 자동으로 수행됩니다.

#### (5) 결과 시각화

```python
cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
```

각 객체마다
- 사각형 박스
- 고유 ID
- 객체별 색상
을 화면에 그립니다.

같은 ID에 같은 색을 유지하도록 `colors` 딕셔너리를 사용하는 점도 중요합니다.
사용자는 박스 색과 번호를 통해 객체가 계속 유지되는지 직관적으로 볼 수 있습니다.

### 8. 이 실습 코드의 전체 의미 정리

이 코드는 하나의 프레임에서 객체를 찾는 **검출(detection)** 과,
여러 프레임에 걸쳐 같은 객체를 계속 식별하는 **추적(tracking)** 을 결합한 예제입니다.

역할을 나누면 다음과 같습니다.

- **YOLOv3**: “현재 프레임에서 무엇이 어디 있는가?”를 찾음
- **SORT**: “방금 검출된 이 객체가 이전 프레임의 누구와 같은가?”를 판단함
- **OpenCV 시각화**: 추적 결과를 사람이 확인할 수 있도록 화면에 표시함

즉, 이 실습의 핵심은 단순히 박스를 그리는 것이 아니라,
**영상 속 객체에 일관된 ID를 부여하여 시간적으로 연결하는 과정**을 이해하는 데 있습니다.


## 실행 방법

```bash
python practice15.py
```

**필수 조건**:
- `yolov3.cfg` 파일: L06 폴더에 위치 ✅
- `yolov3.weights` 파일: L06 폴더에 위치 ✅
- 웹캠: 실시간 입력 필요

**참고 사항**:
- YOLO 모델은 사전 훈련된 80개 클래스를 검출합니다 (COCO 데이터셋)
- 클래스 이름 파일(coco.names) 필요 없음 - 단순히 경계 상자와 ID만 표시합니다
- 초기 로딩 시 weights 파일 크기가 크므로 약간의 시간 소요됨

**키 입력**:
- `ESC` 또는 `Q`: 프로그램 종료

## 실행 결과

프로그램을 실행하면 다음과 같은 화면이 표시됩니다:

```
============================================================
SORT 알고리즘을 활용한 다중 객체 추적기
============================================================

[*] YOLOv3 모델 로딩 중...
[✓] YOLOv3 모델 로딩 완료
[*] 웹캠 연결 중...
[✓] 웹캠 연결 완료

[ 종료: ESC 키 또는 Q 키 ]
============================================================
```

**화면 표시 정보**:
- 각 객체마다 고유한 색상의 경계 상자 표시
- 각 객체의 ID 라벨 (예: ID: 0, ID: 1, ID: 2, ...)
- 현재 프레임 번호
- 추적 중인 객체 수

**추적 성능**:
- 간단하지만 효과적인 실시간 다중 객체 추적
- IoU 기반 매칭으로 안정적 ID 유지
- 프레임 드롭이 발생해도 추적 지속 가능 (max_age 파라미터로 조정)


---

# Practice 16: Mediapipe를 활용한 얼굴 랜드마크 추출 및 시각화

## 전체 코드

```python
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
```

## 주요 코드 설명

### 1. Mediapipe 관련 모듈 초기화 - FaceMesh 기능을 사용하기 위한 준비

```python
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
```

이 부분은 Mediapipe에서 얼굴 랜드마크 검출과 시각화에 필요한 기능들을 꺼내오는 초기 설정입니다.

- `mp_face_mesh`:
  - 얼굴 랜드마크를 검출하는 핵심 모듈입니다.
  - 이 실습에서는 468개의 얼굴 포인트를 찾아냅니다.

- `mp_drawing`:
  - 검출된 랜드마크나 연결선을 이미지 위에 그리기 위한 유틸리티입니다.

- `mp_drawing_styles`:
  - Mediapipe가 제공하는 기본 시각화 스타일을 가져올 때 사용합니다.
  - 예를 들어 메시를 어떤 색과 굵기로 그릴지 정해진 스타일을 쉽게 적용할 수 있습니다.

즉, 이 코드는 “얼굴을 검출하는 엔진”과 “검출 결과를 화면에 그리는 도구”를 각각 준비하는 단계입니다.

### 2. `FaceMesh()` 초기화 - 얼굴 랜드마크 검출기의 동작 방식을 설정

```python
with mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=2,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:
```

이 부분은 실제 얼굴 랜드마크 검출기를 생성하는 핵심 코드입니다.
`with ... as face_mesh:` 형태를 사용했기 때문에,
블록이 끝나면 내부 리소스가 자동으로 정리되는 장점도 있습니다.

각 파라미터의 의미는 다음과 같습니다.

- `static_image_mode=False`
  - 연속된 비디오 프레임을 처리하는 모드입니다.
  - 웹캠처럼 실시간 영상 입력일 때 적합합니다.
  - 한 번 얼굴을 찾은 뒤 다음 프레임에서는 추적 정보를 활용할 수 있어 더 효율적입니다.

- `max_num_faces=2`
  - 동시에 최대 2개의 얼굴까지 검출하도록 설정합니다.
  - 화면에 두 명까지 있어도 각각의 랜드마크를 따로 얻을 수 있습니다.

- `refine_landmarks=True`
  - 기본 랜드마크보다 더 정교한 포인트를 포함합니다.
  - 특히 눈 주변이나 홍채처럼 세밀한 부위를 더 정확하게 잡는 데 도움이 됩니다.

- `min_detection_confidence=0.5`
  - 얼굴이라고 판단할 최소 신뢰도 기준입니다.
  - 값이 높을수록 더 확실한 얼굴만 검출하고, 낮을수록 더 많은 후보를 허용합니다.

- `min_tracking_confidence=0.5`
  - 검출된 얼굴을 다음 프레임에서도 같은 얼굴로 추적할 때 필요한 최소 신뢰도입니다.

정리하면 이 단계는
“몇 명까지 찾을지, 정밀도를 얼마나 높일지, 검출과 추적을 얼마나 엄격하게 할지”를 설정하는 부분입니다.

### 3. `VideoCapture(0)` - 웹캠에서 실시간 영상 입력 받기

```python
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("[✗] 웹캠을 열 수 없습니다")
    return
```

이 부분은 OpenCV를 이용해 기본 웹캠을 여는 코드입니다.

- `0`은 기본 카메라 장치를 뜻합니다.
- 노트북 내장 카메라가 있으면 보통 0번으로 연결됩니다.

`cap.isOpened()`를 바로 확인하는 이유는,
카메라 권한 문제나 장치 연결 오류가 있을 수 있기 때문입니다.
이 검사를 하지 않으면 이후 `cap.read()`에서 오류가 나거나 빈 프레임이 들어올 수 있습니다.

즉, 이 단계는 전체 실습의 입력 데이터를 준비하는 부분입니다.
이 실습에서는 정적인 이미지 파일이 아니라 **실시간 비디오 스트림**을 사용합니다.

### 4. `cap.read()`와 프레임 정보 추출 - 현재 화면 한 장을 가져오기

```python
ret, frame = cap.read()

if not ret:
    print("[✗] 프레임을 읽을 수 없습니다")
    break

height, width, _ = frame.shape
```

웹캠 영상은 사실상 프레임(frame)의 연속입니다.
`cap.read()`를 호출할 때마다 현재 시점의 이미지 한 장을 읽어옵니다.

- `ret`:
  - 프레임을 정상적으로 읽었는지 여부를 나타내는 불리언 값입니다.
- `frame`:
  - 실제 이미지 데이터입니다.

또한 `frame.shape`를 통해 프레임의 크기를 구합니다.

- `height`: 이미지 높이
- `width`: 이미지 너비
- `_`: 채널 수(BGR 3채널)인데 여기서는 직접 사용하지 않아 `_`로 받음

이 정보는 뒤에서 랜드마크 좌표를 픽셀 단위로 변환할 때 꼭 필요합니다.

### 5. BGR → RGB 변환 - OpenCV와 Mediapipe의 색상 순서 차이 맞추기

```python
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
results = face_mesh.process(frame_rgb)
```

이 부분은 실습에서 매우 중요한 전처리 단계입니다.

- OpenCV는 기본적으로 이미지를 **BGR** 순서로 저장합니다.
- 반면 Mediapipe는 입력 이미지를 **RGB** 순서로 기대합니다.

그래서 `cv2.cvtColor()`로 색상 채널 순서를 바꿔줘야 합니다.

이 변환을 하지 않으면,
- 색 정보가 뒤섞인 상태로 모델에 들어가고,
- 얼굴 검출 성능이 떨어질 수 있습니다.

그 다음 `face_mesh.process(frame_rgb)`를 호출하면,
현재 프레임에 얼굴이 있는지 검사하고,
있다면 각 얼굴에 대한 랜드마크 정보를 `results`에 담아 반환합니다.

즉,
- `cvtColor()`는 입력 형식을 맞추는 단계,
- `process()`는 실제 얼굴 랜드마크 추론 단계라고 이해하면 됩니다.

### 6. `results.multi_face_landmarks` - 여러 얼굴의 랜드마크 결과 확인

```python
if results.multi_face_landmarks:
    for face_landmarks in results.multi_face_landmarks:
        num_faces += 1
```

`results.multi_face_landmarks`는 검출된 얼굴들의 랜드마크 목록입니다.

- 얼굴이 하나도 없으면 `None` 또는 비어 있는 상태가 될 수 있습니다.
- 얼굴이 하나 이상 있으면 각 얼굴마다 468개의 랜드마크가 들어 있습니다.

이 코드에서 `for face_landmarks in ...`로 반복하는 이유는,
`max_num_faces=2`로 설정했기 때문에 최대 두 얼굴까지 처리할 수 있기 때문입니다.

즉, 이 루프는
- 첫 번째 얼굴의 랜드마크 처리
- 두 번째 얼굴의 랜드마크 처리
를 차례대로 수행하는 구조입니다.

### 7. `draw_landmarks()` - 얼굴 메시(연결선) 시각화

```python
mp_drawing.draw_landmarks(
    image=frame,
    landmark_list=face_landmarks,
    connections=mp_face_mesh.FACEMESH_TESSELATION,
    landmark_drawing_spec=None,
    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
)
```

이 함수는 검출된 랜드마크들을 서로 연결해 **얼굴 메시(mesh)** 형태로 그립니다.

각 인자의 의미는 다음과 같습니다.

- `image=frame`
  - 결과를 그릴 대상 이미지입니다.
  - 즉, 현재 웹캠 프레임 위에 바로 시각화합니다.

- `landmark_list=face_landmarks`
  - 한 얼굴에 대한 랜드마크 집합입니다.

- `connections=mp_face_mesh.FACEMESH_TESSELATION`
  - 어떤 점과 어떤 점을 연결할지 정의한 목록입니다.
  - 468개 포인트를 삼각형 구조처럼 촘촘히 이어 얼굴 표면 형태를 표현합니다.

- `landmark_drawing_spec=None`
  - 기본 점 그리기는 사용하지 않겠다는 뜻입니다.
  - 여기서는 아래에서 `cv2.circle()`로 직접 점을 그리기 때문입니다.

- `connection_drawing_spec=...`
  - 연결선의 스타일(색상, 두께 등)을 기본 설정으로 적용합니다.

이 단계는 랜드마크를 단순한 점 집합이 아니라,
**얼굴 윤곽과 표면 구조가 보이도록 연결된 메시 형태**로 보여주는 역할을 합니다.

### 8. 랜드마크 좌표 변환 - 정규화 좌표를 실제 픽셀 좌표로 바꾸기

```python
for landmark in face_landmarks.landmark:
    x = int(landmark.x * width)
    y = int(landmark.y * height)
    cv2.circle(frame, (x, y), 1, (255, 255, 0), -1)
```

Mediapipe가 반환하는 `landmark.x`, `landmark.y`는 픽셀 좌표가 아니라
**0~1 범위로 정규화된 좌표**입니다.

즉,
- `landmark.x = 0.5`는 이미지 너비의 중앙
- `landmark.y = 0.25`는 이미지 높이의 25% 지점
을 의미합니다.

그래서 실제 화면에 점을 찍으려면 다음과 같이 변환해야 합니다.

$$
 x_{pixel} = x_{normalized} 	imes width
$$
$$
 y_{pixel} = y_{normalized} 	imes height
$$

코드에서는 이 값을 `int()`로 정수 픽셀 좌표로 바꾸고,
`cv2.circle()`로 반지름 1짜리 작은 점을 그립니다.

여기서 사용한 `(255, 255, 0)`은 OpenCV의 BGR 순서 기준 색상입니다.
즉, 화면에는 노란색 계열 점으로 랜드마크가 표시됩니다.

이 단계 덕분에 얼굴 메시뿐 아니라 각 랜드마크의 정확한 위치를 개별 점으로도 확인할 수 있습니다.

### 9. 프레임 정보 출력 - 현재 처리 상황을 화면에 표시

```python
info_text = f"Frame: {frame_count} | Detected Faces: {num_faces}"
cv2.putText(frame, info_text, (10, 30),
           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
```

이 코드는 현재 프레임 번호와 검출된 얼굴 수를 화면 왼쪽 위에 출력합니다.

- `frame_count`:
  - 지금까지 몇 번째 프레임을 처리 중인지 보여줍니다.
- `num_faces`:
  - 현재 프레임에서 몇 개의 얼굴이 검출되었는지 보여줍니다.

이 정보는 디버깅이나 실시간 처리 상태 확인에 유용합니다.
예를 들어,
- 얼굴이 잘 잡히는지,
- 두 명 이상이 동시에 잘 인식되는지,
- 프레임이 계속 정상적으로 갱신되는지
를 한눈에 볼 수 있습니다.

### 10. `cv2.imshow()`와 `waitKey()` - 실시간 출력과 종료 처리

```python
cv2.imshow('Face Landmarks Detection with Mediapipe', frame)

key = cv2.waitKey(1) & 0xFF
if key == 27:
    break
```

- `cv2.imshow()`:
  - 가공이 끝난 프레임을 창에 띄워 사용자가 볼 수 있게 합니다.

- `cv2.waitKey(1)`:
  - 1ms 동안 키 입력을 기다립니다.
  - 동시에 화면을 갱신하는 역할도 합니다.

- `27`:
  - ESC 키의 ASCII 코드입니다.
  - 사용자가 ESC를 누르면 반복문을 종료하여 프로그램을 끝냅니다.

실시간 비디오 프로그램에서는 이 패턴이 매우 자주 사용됩니다.
`imshow()`만 있고 `waitKey()`가 없으면 창이 제대로 갱신되지 않을 수 있습니다.

### 11. 리소스 해제 - 프로그램 종료 시 카메라와 창 정리

```python
cap.release()
cv2.destroyAllWindows()
```

실습이 끝나면 사용한 리소스를 반드시 정리해야 합니다.

- `cap.release()`:
  - 웹캠 장치 사용을 종료합니다.
  - 이 작업을 하지 않으면 다른 프로그램이 카메라를 못 쓸 수 있습니다.

- `cv2.destroyAllWindows()`:
  - OpenCV로 띄운 모든 창을 닫습니다.

즉, 이 코드는 단순히 프로그램을 끝내는 것이 아니라,
운영체제 수준에서 점유 중이던 장치와 창 자원을 깔끔하게 반환하는 마무리 단계입니다.

### 12. 이 실습 코드의 전체 의미 정리

이 실습은 단순히 얼굴을 네모 박스로 찾는 것이 아니라,
얼굴의 세부 구조를 이루는 **468개의 랜드마크를 실시간으로 추출하고 시각화**하는 예제입니다.

전체 흐름은 다음과 같이 이해하면 됩니다.

1. 웹캠에서 프레임을 읽는다.
2. OpenCV의 BGR 이미지를 Mediapipe용 RGB 이미지로 바꾼다.
3. FaceMesh가 얼굴과 468개 랜드마크를 검출한다.
4. 각 랜드마크를 선과 점으로 그려 얼굴 구조를 시각화한다.
5. 프레임 번호와 얼굴 수를 출력하며 실시간으로 반복한다.

즉, 이 코드는 얼굴 인식의 다음 단계인
**얼굴 자세 분석, 표정 인식, 시선 추정, 얼굴 3D 구조 이해** 같은 응용의 기초가 되는 랜드마크 추출 과정을 직접 보여주는 실습이라고 볼 수 있습니다.


## Mediapipe FaceMesh의 특징

### 468개 얼굴 랜드마크 구성

- **입술**: 약 20개 포인트
- **왼쪽 눈**: 약 31개 포인트
- **오른쪽 눈**: 약 31개 포인트
- **코**: 약 10개 포인트
- **턱과 얼굴 외곽**: 약 376개 포인트

### FACEMESH_TESSELATION

메시 구조로 얼굴의 3D 표면을 삼각형으로 연결:
```python
mp_face_mesh.FACEMESH_TESSELATION  # 468개 포인트를 연결하는 삼각형 메시
```

## 실행 방법

```bash
python practice16.py
```

**필수 조건**:
- mediapipe 설치: `pip install mediapipe`
- opencv-python 설치: `pip install opencv-python`
- 웹캠: 실시간 입력 필요

**키 입력**:
- `ESC`: 프로그램 종료

## 실행 결과

프로그램을 실행하면 다음과 같은 화면이 표시됩니다:

```
============================================================
Mediapipe를 활용한 얼굴 랜드마크 추출 및 시각화
============================================================

[*] 웹캠 연결 중...
[✓] 웹캠 연결 완료

[*] Mediapipe FaceMesh 모델 로딩 중...
[✓] 모델 로딩 완료

[ 종료: ESC 키 ]
============================================================
```

**화면 표시 정보**:
- **흰색 선**: 얼굴 메시 (468개 포인트를 잇는 삼각형)
- **노란색 점**: 개별 랜드마크
- **상단 정보**: 
  - 현재 프레임 번호
  - 검출된 얼굴 수

**화면 예시**:
```
Frame: 45 | Detected Faces: 1
[얼굴 메시와 랜드마크 점들이 실시간으로 표시됨]
```

