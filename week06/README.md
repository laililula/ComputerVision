# Week 06: SORT 알고리즘을 이용한 다중 객체 추적

## Practice 15: SORT 알고리즘을 활용한 실시간 다중 객체 추적

### 개요

이 실습은 **YOLOv3 객체 검출**과 **SORT(Simple Online and Realtime Tracking) 알고리즘**을 결합하여 실시간 비디오 스트림에서 **여러 객체를 동시에 추적**하는 프로젝트입니다.

검출(Detection): 각 프레임에서 객체의 위치를 찾는 단계
추적(Tracking): 프레임들 간에 같은 객체인지를 판단하고 고유 ID를 유지하는 단계

SORT 알고리즘은 간단하면서도 효과적인 추적 기법으로, IoU(Intersection over Union) 기반의 매칭과 Hungarian 알고리즘을 사용합니다.

## 전체 코드

```python
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

# ==========================================
# 1. IoU 계산 함수
# ==========================================

def iou(bb_test, bb_gt):
    """
    두 바운딩 박스 간의 IoU(Intersection over Union) 계산.
    bb_test, bb_gt: [x1, y1, x2, y2] 형식의 박스 좌표
    반환: 0 ~ 1 사이의 IoU 값
    """
    xx1 = max(bb_test[0], bb_gt[0])
    yy1 = max(bb_test[1], bb_gt[1])
    xx2 = min(bb_test[2], bb_gt[2])
    yy2 = min(bb_test[3], bb_gt[3])
    
    w = max(0., xx2 - xx1)
    h = max(0., yy2 - yy1)
    
    inter = w * h
    
    area1 = (bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
    area2 = (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1])
    union = area1 + area2 - inter
    
    if union <= 0:
        return 0.0
    
    return inter / union


# ==========================================
# 2. Track 클래스 - 개별 객체 관리
# ==========================================

class Track:
    """개별 추적 객체를 관리하는 클래스"""
    
    def __init__(self, bbox, track_id):
        """
        bbox: [x1, y1, x2, y2] 형식의 초기 바운딩 박스
        track_id: 이 객체의 고유 ID
        """
        self.bbox = bbox           # 현재 박스 좌표
        self.id = track_id         # 객체의 고유 ID (변하지 않음)
        self.hits = 1              # 성공적으로 detection과 매칭된 횟수
        self.no_losses = 0         # 연속으로 매칭되지 않은 프레임 수
    
    def update(self, bbox):
        """새로운 detection과 매칭되었을 때 호출"""
        self.bbox = bbox           # 박스 위치 업데이트
        self.hits += 1             # 성공 카운트 증가
        self.no_losses = 0         # 손실 카운트 리셋


# ==========================================
# 3. SimpleSORT 클래스 - SORT 알고리즘 구현
# ==========================================

class SimpleSORT:
    """간단하지만 효과적인 SORT 알고리즘 구현"""
    
    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
        """
        max_age: 최대 몇 프레임까지 detection 없이 track 유지할지
        min_hits: 화면에 표시하기 위한 최소 매칭 횟수
        iou_threshold: 유효한 매칭으로 간주할 최소 IoU 값
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks = []           # 활성 track 목록
        self.next_id = 0           # 다음 부여할 ID
    
    def update(self, detections):
        """
        현재 프레임의 detection 결과를 받아 track을 업데이트하고,
        현재 추적 중인 객체 목록을 반환.
        
        detections: [[x1, y1, x2, y2], ...] 형식의 bbox 목록
        반환: [[x1, y1, x2, y2, id, hits], ...] 형식의 추적 결과
        """
        
        # ========== 단계 1: IoU 행렬 계산 ==========
        # 기존 track들과 새로운 detection들 사이의 IoU 계산
        iou_matrix = np.zeros((len(self.tracks), len(detections)), dtype=np.float32)
        for t, trk in enumerate(self.tracks):
            for d, det in enumerate(detections):
                iou_matrix[t, d] = iou(trk.bbox, det[:4])
        
        # ========== 단계 2: Hungarian 알고리즘으로 최적 매칭 ==========
        # IoU를 비용으로 변환 (음수 = 최소화 문제가 최대화 문제가 되도록)
        row_ind, col_ind = linear_sum_assignment(-iou_matrix)
        
        # ========== 단계 3: 매칭 결과 처리 ==========
        assigned_tracks = set()
        assigned_dets = set()
        
        for r, c in zip(row_ind, col_ind):
            # IoU가 임계값 이상인 경우만 유효한 매칭으로 간주
            if iou_matrix[r, c] >= self.iou_threshold:
                self.tracks[r].update(detections[c][:4])
                assigned_tracks.add(r)
                assigned_dets.add(c)
        
        # ========== 단계 4: 미매칭 track 처리 ==========
        # 현재 프레임에서 detection과 매칭되지 않은 track
        for t, trk in enumerate(self.tracks):
            if t not in assigned_tracks:
                trk.no_losses += 1  # 손실 프레임 증가
        
        # ========== 단계 5: 미매칭 detection 처리 ==========
        # 기존 track과 매칭되지 않은 새 detection
        for d, det in enumerate(detections):
            if d not in assigned_dets:
                self.tracks.append(Track(det[:4], self.next_id))
                self.next_id += 1
        
        # ========== 단계 6: 오래된 track 제거 ==========
        # max_age보다 오래 감지되지 않은 track 제거
        self.tracks = [t for t in self.tracks if t.no_losses <= self.max_age]
        
        # ========== 결과 반환 ==========
        # min_hits 이상 매칭되고 현재 프레임에서 업데이트된 track만 반환
        results = []
        for t in self.tracks:
            if t.no_losses == 0 and t.hits >= self.min_hits:
                results.append([t.bbox[0], t.bbox[1], t.bbox[2], t.bbox[3], t.id, t.hits])
        
        return results


# ==========================================
# 4. YOLOv3 설정 및 초기화
# ==========================================

# YOLOv3 모델 파일 경로
config_path = '../L06/yolov3.cfg'
weights_path = '../L06/yolov3.weights'

# 차량 클래스 ID (COCO): car=2, motorbike=3, bus=5, truck=7
vehicle_ids = {2, 3, 5, 7}

# 네트워크 로드
net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# 출력층 이름 가져오기
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# 웹캠 열기
cap = cv2.VideoCapture(0)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# 추적기 초기화
sort_tracker = SimpleSORT(max_age=30, min_hits=3, iou_threshold=0.3)

# 객체별 색상 딕셔너리
colors = {}

print("=" * 60)
print("SORT 알고리즘을 활용한 다중 객체 추적기")
print("=" * 60)
print(f"\n[*] YOLOv3 모델 로딩 중...")
print(f"[✓] Config: {config_path}")
print(f"[✓] Weights: {weights_path}")
print(f"[*] 웹캠 연결 중...")
print(f"[✓] 웹캠 해상도: {frame_width} x {frame_height}, FPS: {fps}")
print(f"\n[ 종료: ESC 키 또는 Q 키 ]")
print("=" * 60)

frame_count = 0

# ==========================================
# 5. 메인 루프 - 프레임 처리
# ==========================================

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    
    # 검출 속도 향상을 위해 프레임 리사이즈
    resized_frame = cv2.resize(frame, (640, 480))
    height_resized, width_resized = resized_frame.shape[:2]
    
    # 원본 좌표로 복원하기 위한 스케일 계산
    scale_x = frame_width / width_resized
    scale_y = frame_height / height_resized
    
    # ====== YOLOv3 검출 ======
    blob = cv2.dnn.blobFromImage(resized_frame, 1/255.0, (416, 416), 
                                  swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    
    # 검출 결과 파싱
    boxes = []
    confidences = []
    
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            # 차량 클래스이고 신뢰도가 0.5 이상인 경우만 선택
            if class_id in vehicle_ids and confidence > 0.5:
                center_x = int(detection[0] * width_resized)
                center_y = int(detection[1] * height_resized)
                w = int(detection[2] * width_resized)
                h = int(detection[3] * height_resized)
                
                # 중심 좌표 → 모서리 좌표 변환
                x1 = max(0, center_x - w // 2)
                y1 = max(0, center_y - h // 2)
                x2 = min(width_resized - 1, center_x + w // 2)
                y2 = min(height_resized - 1, center_y + h // 2)
                
                boxes.append([x1, y1, x2, y2])
                confidences.append(float(confidence))
    
    # NMS (중복 박스 제거)
    if boxes:
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        detections = []
        for i in indices:
            box = boxes[i]
            # 원본 프레임 좌표로 복원
            x1_orig = int(box[0] * scale_x)
            y1_orig = int(box[1] * scale_y)
            x2_orig = int(box[2] * scale_x)
            y2_orig = int(box[3] * scale_y)
            detections.append([x1_orig, y1_orig, x2_orig, y2_orig])
    else:
        detections = []
    
    # ====== SORT 추적기 업데이트 ======
    tracked_objects = sort_tracker.update(detections)
    
    # ====== 결과 시각화 ======
    for obj in tracked_objects:
        x1, y1, x2, y2, obj_id, hits = obj
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # 객체 ID별 색상 할당 (없으면 새로 할당)
        if obj_id not in colors:
            colors[obj_id] = tuple(np.random.randint(0, 256, 3).tolist())
        
        color = colors[obj_id]
        
        # 바운딩 박스 그리기
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # ID 라벨 그리기
        label = f"ID: {obj_id}"
        cv2.putText(frame, label, (x1, y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # 프레임 번호 표시
    cv2.putText(frame, f"Frame: {frame_count}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Objects: {len(tracked_objects)}", (10, 70),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # 결과 출력
    cv2.imshow('SORT Object Tracking', frame)
    
    # 키 입력 처리
    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == ord('q'):  # ESC 또는 Q
        break

# 비디오 파일 닫기
cap.release()

# OpenCV 창 모두 닫기
cv2.destroyAllWindows()
```

## 주요 코드 설명

### 1. `iou()` 함수 - 두 바우딩 박스의 겹침 정도를 계산

```python
def iou(bb_test, bb_gt):
    """두 바운딩 박스 간의 IoU(Intersection over Union) 계산"""
    xx1 = max(bb_test[0], bb_gt[0])
    yy1 = max(bb_test[1], bb_gt[1])
    xx2 = min(bb_test[2], bb_gt[2])
    yy2 = min(bb_test[3], bb_gt[3])
    
    w = max(0., xx2 - xx1)
    h = max(0., yy2 - yy1)
    
    inter = w * h
    
    area1 = (bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
    area2 = (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1])
    union = area1 + area2 - inter
    
    if union <= 0:
        return 0.0
    
    return inter / union
```

**목적**: 이전 프레임의 추적 박스와 현재 프레임의 검출 박스가 얼마나 겹치는지를 정량화합니다.

**공식**:
$$IoU = \frac{\text{교집합 면적}}{\text{합집합 면적}}$$

**해석**:
- IoU = 1: 두 박스가 완전히 동일
- IoU = 0: 두 박스가 전혀 겹치지 않음
- 0 < IoU < 1: 부분적으로 겹침

SORT 알고리즘에서는 이 값을 기준으로 같은 객체인지를 판단합니다. 일반적으로 IoU > 0.3~0.5를 만족하면 같은 객체로 매칭합니다.

### 2. `Track` 클래스 - 개별 추적 객체의 생명주기 관리

```python
class Track:
    def __init__(self, bbox, track_id):
        self.bbox = bbox           # [x1, y1, x2, y2]: 현재 바운딩 박스
        self.id = track_id         # 고유 ID (변하지 않음)
        self.hits = 1              # 성공적 매칭 횟수 (초기값 1)
        self.no_losses = 0         # 미매칭 프레임 수 (초기값 0)
    
    def update(self, bbox):
        """새로운 detection과 매칭되었을 때 호출"""
        self.bbox = bbox           # 박스 위치 갱신
        self.hits += 1             # 성공 카운트 증가
        self.no_losses = 0         # 손실 카운트 초기화
```

**목적**: SORT 추적기에서 관리하는 **개별 객체 1개를 표현하는 단위**입니다.

**속성 상세 설명**:

- `bbox`: 현재 프레임에서 이 객체의 위치
  - SORT.update() 호출 시 새 detection으로 계속 갱신됨

- `id`: 객체의 고유 식별자
  - 프로그램 실행 중 절대 변하지 않음
  - 사용자가 객체를 시각적으로 추적할 때 사용

- `hits`: 연속 성공 매칭 횟수
  - detection과 매칭될 때마다 증가
  - `min_hits` 파라미터와 비교하여 추적 안정성 판단
  - 예: `min_hits=3`이면 3번 이상 매칭되어야 화면에 표시

- `no_losses`: 연속 미매칭 프레임 수
  - 매칭 실패 시 증가, 성공 시 0으로 초기화
  - `max_age` 파라미터와 비교하여 객체 제거 판단
  - 예: `max_age=10`이면 10프레임 이상 감지 안 되면 삭제

### 3. `SimpleSORT.update()` - SORT 알고리즘의 6단계 핵심 로직

```python
def update(self, detections):
    """
    현재 프레임의 detection들을 받아서:
    1. 기존 track들과 IoU 행렬 계산
    2. Hungarian 알고리즘으로 최적 매칭
    3. 매칭 결과 반영 및 새 track 생성
    4. 오래된 track 제거
    """
    
    # 단계 1: IoU 행렬 계산 (track 수 × detection 수)
    iou_matrix = np.zeros((len(self.tracks), len(detections)), dtype=np.float32)
    for t, trk in enumerate(self.tracks):
        for d, det in enumerate(detections):
            iou_matrix[t, d] = iou(trk.bbox, det[:4])
    
    # 단계 2: Hungarian 알고리즘으로 최적 일대일 매칭
    row_ind, col_ind = linear_sum_assignment(-iou_matrix)
```

**SORT 6단계 상세 해설**:

**단계 1 & 2: IoU 행렬과 Hungarian 매칭**
- `iou_matrix[t, d]` = t번 track과 d번 detection의 IoU 값
- `linear_sum_assignment(-iou_matrix)`:
  - Hungarian 알고리즘은 비용 최소화 문제를 풀기 때문에 `-iou_matrix` 사용
  - IoU가 크면 비용이 작아져서 높은 IoU 매칭이 우선됨
  - 결과: 전체 관점에서 최적의 일대일 매칭 조합

**Hungarian이 필요한 이유**:
단순하게 각 track이 최고 IoU의 detection을 택하면:
  - 여러 track이 같은 detection 선택 가능 (중복)
  - 전체가 비효율적 (예: track1에 IoU 0.9, track2에 IoU 0.8인 detection이 있을 때, 단순 방식은 각각 자기 최고를 택하지만, Hungarian은 전체 합이 1.7이 되도록 조정)

**단계 3: 매칭 결과 반영**
```python
for r, c in zip(row_ind, col_ind):
    if iou_matrix[r, c] >= self.iou_threshold:  # 예: 0.3
        self.tracks[r].update(detections[c][:4])
        assigned_dets.add(c)
    # else: 임계값 미만은 매칭 무시
```
- IoU > threshold인 경우만 유효한 매칭으로 간주
- 임계값 미만 매칭은 강제 연결을 방지하기 위해 무시

**단계 4: 미매칭 track 처리**
```python
for t, trk in enumerate(self.tracks):
    if t not in assigned_tracks:
        trk.no_losses += 1  # 손실 프레임 증가
```
- 현재 프레임에서 detection과 짝지어지지 않은 track
- 의미: "객체가 잠시 가려짐" 또는 "검출기가 놓침"
- 대응: `no_losses` 증가하여 `max_age` 동안만 유지

**단계 5: 미래칭 detection 처리**
```python
for d, det in enumerate(detections):
    if d not in assigned_dets:
        self.tracks.append(Track(det[:4], self.next_id))
        self.next_id += 1
```
- 어떤 기존 track과도 연결되지 않은 새 detection
- 의미: **화면에 처음 등장한 객체**
- 대응: 새로운 `Track` 객체 생성 및 새 ID 부여

**단계 6: 오래된 track 제거**
```python
self.tracks = [t for t in self.tracks if t.no_losses <= self.max_age]
```
- `no_losses > max_age` track은 화면에서 영구히 사라진 것 간주
- 메모리 관리 및 ID 재사용 목적

**이 설계의 장점**:
- 안정성: 짧은 검출 실패(1~2프레임)에도 ID 유지
- 자동성: 새 객체 자동 인식, 사라진 객체 자동 제거
- 효율성: 전체 최적화를 통한 ID 갈등 최소화

### 4. YOLOv3 검출 파이프라인 - blob 생성부터 NMS까지

```python
# 1. 이미지 → blob 변환
blob = cv2.dnn.blobFromImage(
    resized_frame, 1/255.0, (416, 416), 
    swapRB=True, crop=False
)

# 2. 네트워크 입력 및 추론  
net.setInput(blob)
outs = net.forward(output_layers)

# 3. 검출 결과 파싱 및 필터링
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        
        if class_id in vehicle_ids and confidence > 0.5:
            # 중심 좌표 → 모서리 좌표 변환
            center_x = int(detection[0] * width_resized)
            center_y = int(detection[1] * height_resized)
            w = int(detection[2] * width_resized)
            h = int(detection[3] * height_resized)
            
            x1 = max(0, int(center_x - w / 2))
            y1 = max(0, int(center_y - h / 2))
            x2 = min(width_resized - 1, int(center_x + w / 2))
            y2 = min(height_resized - 1, int(center_y + h / 2))
            
            boxes.append([x1, y1, x2, y2])
            confidences.append(float(confidence))

# 4. NMS (중복 박스 제거)
indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
```

**파이프라인 4단계 상세 설명**:

**단계 1: Blob 변환 (전처리)**
- `1/255.0`: 픽셀값 0~255→0~1 정규화
  - 신경망 학습 시 입력 범위를 맞춰주어 안정성 증대
- `(416, 416)`: YOLO 정확한 입력 크기
  - 이 크기가 아니면 네트워크 에러 발생
- `swapRB=True`: BGR→RGB 색상 순서 변환
  - OpenCV는 BGR, YOLO는 RGB 권장
- `crop=False`: 비율 유지하며 resize (왜곡 방지)

**단계 2: 추론**
- `net.forward(output_layers)` 호출로 신경망 정방향 전파
- YOLO는 멀티스케일 검출 (작은 객체~큰 객체 동시 감지)
- 반환값: 여러 출력층에서의 검출 후보들

**단계 3: 후보 필터링 및 좌표 변환**

각 detection 구조: `[cx, cy, w, h, objectness, class1_score, ..., class80_score]`

- `scores = detection[5:]`: 80개 클래스 점수
- `class_id = np.argmax(scores)`: 최고 점수 클래스만 선택
- 조건 필터링:
  - `class_id in vehicle_ids`: COCO에서 차량류만 추출 (car=2, bike=3, bus=5, truck=7)
  - `confidence > 0.5`: 신뢰도 50% 이상만 유지

**좌표 변환**: 중심 → 모서리 형식
- 이유: IoU 계산과 시각화에 `[x1, y1, x2, y2]` 형식 필요
- `x1 = center_x - w/2` (좌상단 x)
- `y1 = center_y - h/2` (좌상단 y)
- `x2 = center_x + w/2` (우하단 x)
- `y2 = center_y + h/2` (우하단 y)
- 경계 보정: 좌표가 이미지 범위를 벗어나지 않도록 clamp

**단계 4: NMS (Non-Maximum Suppression)**
- **문제**: 같은 객체에 여러 박스 예측 (예: 겹치는 박스들)
- **해결**: IoU 기준으로 중복 제거
- `nms_threshold=0.4`: IoU > 0.4인 박스들 중 신뢰도 낮은 것 삭제
- 결과: 중복 없는 고품질 최종 검출 결과

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
[✓] Config: ../L06/yolov3.cfg
[✓] Weights: ../L06/yolov3.weights
[*] 웹캠 연결 중...
[✓] 웹캠 해상도: 1280 x 720, FPS: 30.0

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
import cv2
import mediapipe as mp
import numpy as np

# MediaPipe 초기화
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# 웹캠 설정
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 웹캠 연결
cap = cv2.VideoCapture(0)

print("=" * 60)
print("Mediapipe 얼굴 랜드마크 추출")
print("=" * 60)
print("[ 종료: ESC 키 또는 Q 키 ]")
print("=" * 60)

frame_count = 0

# 메인 루프
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    
    # RGB 변환 (MediaPipe는 RGB 입력 요구)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 얼굴 랜드마크 검출
    results = face_mesh.process(rgb_frame)
    
    # 결과 시각화
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # 랜드마크 좌표 추출
            h, w, c = frame.shape
            
            # 주요 랜드마크만 강조해서 그리기
            # 눈 (index 33, 133), 코 (34), 입 (78, 308)
            key_points = [33, 133, 34, 78, 308]
            
            for idx in key_points:
                landmark = face_landmarks.landmark[idx]
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
            
            # 전체 랜드마크 그리기
            mp_drawing.draw_landmarks(
                frame,
                face_landmarks,
                mp_face_mesh.FACEMESH_CONTOURS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=1)
            )
    
    # 프레임 번호 표시
    cv2.putText(frame, f"Frame: {frame_count}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # 결과 출력
    cv2.imshow('Face Mesh', frame)
    
    # 키 입력 처리
    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == ord('q'):  # ESC 또는 Q
        break

# 리소스 해제
cap.release()
cv2.destroyAllWindows()
face_mesh.close()
```

## 주요 코드 설명

### 1. MediaPipe FaceMesh 초기화

```python
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
```

- `static_image_mode=False`: 동영상 모드 (추적 활성화)
- `max_num_faces=2`: 최대 2명의 얼굴 동시 처리
- `min_detection_confidence=0.5`: 검출 신뢰도 50% 이상
- `min_tracking_confidence=0.5`: 추적 신뢰도 50% 이상

### 2. 프레임 처리 및 랜드마크 검출

```python
# RGB 변환 (MediaPipe는 RGB 입력 요구)
rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# 얼굴 랜드마크 검출
results = face_mesh.process(rgb_frame)
```

- OpenCV의 BGR 이미지를 RGB로 변환
- FaceMesh 모델로 얼굴 랜드마크 468개 검출

### 3. 랜드마크 시각화

```python
if results.multi_face_landmarks:
    for face_landmarks in results.multi_face_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            face_landmarks,
            mp_face_mesh.FACEMESH_CONTOURS
        )
```

- 검출된 각 얼굴의 랜드마크를 메시 형태로 그리기
- FACEMESH_CONTOURS: 얼굴의 윤곽선 연결 정보

## 실행 방법

```bash
python practice16.py
```

**필수 라이브러리**:
```bash
pip install mediapipe
```

**실행 결과**:
- 웹캠에서 얼굴을 인식하고 468개의 랜드마크 표시
- 각 랜드마크가 연결되어 3D 얼굴 메시 형태로 시각화
- 실시간으로 얼굴 움직임 추적

---
