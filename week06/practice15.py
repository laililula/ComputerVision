# OpenCV: 영상 읽기, DNN 객체 검출, 화면 출력에 사용
import cv2

# NumPy: 배열 처리, IoU 행렬 생성, 검출 결과 저장에 사용
import numpy as np

# Hungarian Algorithm: tracker와 detection을 최적으로 매칭하는 데 사용
from scipy.optimize import linear_sum_assignment


# =========================
# 1. IoU 계산 함수
# =========================

# 두 개의 바운딩 박스(bb_test, bb_gt)가 얼마나 겹치는지를
# IoU(Intersection over Union) 값으로 계산하는 함수
def iou(bb_test, bb_gt):
    # 두 박스가 겹치는 영역의 좌상단 x, y
    xx1 = max(bb_test[0], bb_gt[0])
    yy1 = max(bb_test[1], bb_gt[1])

    # 두 박스가 겹치는 영역의 우하단 x, y
    xx2 = min(bb_test[2], bb_gt[2])
    yy2 = min(bb_test[3], bb_gt[3])

    # 겹치는 영역의 너비와 높이
    w = max(0., xx2 - xx1)
    h = max(0., yy2 - yy1)

    # 교집합 면적
    inter = w * h

    # 첫 번째 박스의 면적
    area1 = (bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])

    # 두 번째 박스의 면적
    area2 = (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1])

    # 합집합 면적
    union = area1 + area2 - inter

    # 합집합이 0 이하이면 나눗셈 오류 방지를 위해 0 반환
    if union <= 0:
        return 0.0

    # IoU = 교집합 / 합집합
    return inter / union


# =========================
# 2. 간단한 SORT용 Track 클래스
# =========================

# 하나의 추적 객체(트랙)를 표현하는 클래스
# bbox: 현재 객체의 바운딩 박스
# id: 객체 고유 번호
# hits: 지금까지 성공적으로 매칭된 횟수
# no_losses: 최근 몇 프레임 동안 검출과 매칭되지 않았는지
class Track:
    def __init__(self, bbox, track_id):
        # 현재 객체의 위치 정보 [x1, y1, x2, y2]
        self.bbox = bbox

        # 이 객체의 고유 ID
        self.id = track_id

        # 첫 생성 시 매칭 횟수는 1
        self.hits = 1

        # 아직 잃어버린 적 없으므로 0
        self.no_losses = 0

    # 기존 track이 새 detection과 매칭되었을 때 정보 갱신
    def update(self, bbox):
        # 바운딩 박스를 새 위치로 갱신
        self.bbox = bbox

        # 성공적으로 다시 매칭되었으므로 hits 증가
        self.hits += 1

        # 놓친 프레임 수 초기화
        self.no_losses = 0


# =========================
# 3. 간단한 SORT 클래스
# =========================

# SORT의 핵심 아이디어를 단순화한 추적기 클래스
# 원본 SORT처럼 Kalman Filter는 사용하지 않고,
# IoU 기반 매칭과 생존 시간 관리만으로 구현
class SimpleSORT:
    def __init__(self, max_age=10, min_hits=3, iou_threshold=0.3):
        # 몇 프레임까지 매칭이 안 되어도 track을 유지할지
        self.max_age = max_age

        # 몇 번 이상 매칭된 객체만 안정적인 track으로 볼지
        self.min_hits = min_hits

        # tracker와 detection을 같은 객체로 볼 최소 IoU 기준
        self.iou_threshold = iou_threshold

        # 현재 살아있는 track 목록
        self.tracks = []

        # 새 객체에 부여할 다음 ID 번호
        self.next_id = 1

    # 현재 프레임의 detections를 받아 tracks를 업데이트하는 함수
    def update(self, detections):
        """
        detections: [[x1, y1, x2, y2, conf], ...]
        return: [[x1, y1, x2, y2, track_id], ...]
        """

        # 최종적으로 화면에 표시할 추적 결과 저장
        updated_tracks = []

        # 아직 track이 하나도 없으면
        # 현재 detection들을 전부 새 객체로 등록
        if len(self.tracks) == 0:
            for det in detections:
                bbox = det[:4]
                self.tracks.append(Track(bbox, self.next_id))
                self.next_id += 1

        else:
            # detection이 하나 이상 있으면 매칭 수행
            if len(detections) > 0:
                # tracker 수 x detection 수 크기의 IoU 행렬 생성
                iou_matrix = np.zeros((len(self.tracks), len(detections)), dtype=np.float32)

                # 각 tracker와 detection 쌍의 IoU 계산
                for t, trk in enumerate(self.tracks):
                    for d, det in enumerate(detections):
                        iou_matrix[t, d] = iou(trk.bbox, det[:4])

                # Hungarian Algorithm을 사용하기 위해
                # IoU를 최대화하는 문제를 -IoU 최소화 문제로 바꿔서 계산
                row_ind, col_ind = linear_sum_assignment(-iou_matrix)

                # 매칭된 tracker와 detection 인덱스를 저장할 집합
                assigned_tracks = set()
                assigned_dets = set()

                # Hungarian 결과 중에서 IoU가 threshold 이상인 것만 진짜 매칭으로 인정
                for r, c in zip(row_ind, col_ind):
                    if iou_matrix[r, c] >= self.iou_threshold:
                        self.tracks[r].update(detections[c][:4])
                        assigned_tracks.add(r)
                        assigned_dets.add(c)

                # 매칭되지 못한 tracker는 한 프레임 놓친 것으로 처리
                for t, trk in enumerate(self.tracks):
                    if t not in assigned_tracks:
                        trk.no_losses += 1

                # 매칭되지 못한 detection은 새 객체로 등록
                for d, det in enumerate(detections):
                    if d not in assigned_dets:
                        self.tracks.append(Track(det[:4], self.next_id))
                        self.next_id += 1

            else:
                # detection이 하나도 없으면 모든 기존 track이 한 프레임씩 손실
                for trk in self.tracks:
                    trk.no_losses += 1

        # 너무 오래 매칭되지 않은 track은 제거
        self.tracks = [t for t in self.tracks if t.no_losses <= self.max_age]

        # 화면 출력용 결과 생성
        # 충분히 안정적으로 검출된 객체이거나,
        # 방금 막 검출된 객체는 출력 대상에 포함
        for trk in self.tracks:
            if trk.hits >= self.min_hits or trk.no_losses == 0:
                x1, y1, x2, y2 = trk.bbox
                updated_tracks.append([int(x1), int(y1), int(x2), int(y2), trk.id])

        # 최종 추적 결과 반환
        return updated_tracks


# =========================
# 4. YOLOv3 파일 경로 설정
# =========================

# YOLO 가중치 파일 경로
weights_path = "L06/yolov3.weights"

# YOLO 설정 파일 경로
config_path = "L06/yolov3.cfg"

# 추적할 입력 비디오 경로
video_path = "L06/slow_traffic_small.mp4"


# =========================
# 5. YOLOv3 네트워크 로드
# =========================

# cfg와 weights 파일을 이용해 YOLOv3 네트워크 생성
net = cv2.dnn.readNetFromDarknet(config_path, weights_path)

# 네트워크의 전체 레이어 이름 가져오기
layer_names = net.getLayerNames()

# YOLO의 최종 출력 레이어 이름만 추출
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]


# =========================
# 6. 추적할 차량 클래스 지정
# =========================

# COCO 데이터셋 기준 클래스 번호
# car=2, motorbike=3, bus=5, truck=7
# 여기서는 차량류만 추적하도록 설정
vehicle_ids = [2, 3, 5, 7]


# =========================
# 7. 비디오 열기 + SORT 초기화
# =========================

# 입력 비디오 파일 열기
cap = cv2.VideoCapture(video_path)

# 비디오가 열리지 않았으면 오류 메시지 출력 후 종료
if not cap.isOpened():
    print("비디오를 열 수 없습니다.")
    exit()

# 추적기 생성
# max_age=10: 10프레임까지 안 보여도 유지
# min_hits=2: 2번 이상 검출되면 안정적인 객체로 판단
# iou_threshold=0.3: 매칭 기준 IoU
tracker = SimpleSORT(max_age=10, min_hits=2, iou_threshold=0.3)


# =========================
# 8. 프레임 단위 처리 시작
# =========================

# 비디오를 한 프레임씩 끝까지 읽기
while True:
    # 현재 프레임 읽기
    ret, frame = cap.read()

    # 더 이상 읽을 프레임이 없으면 종료
    if not ret:
        break

    # 현재 프레임의 높이와 너비 얻기
    height, width = frame.shape[:2]


    # =========================
    # 9. YOLO 입력용 blob 생성
    # =========================

    # 이미지를 YOLO 입력 형식으로 변환
    # 1/255로 정규화
    # 416x416 크기로 resize
    # swapRB=True: OpenCV BGR -> RGB 순서 교체
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)

    # blob을 네트워크 입력으로 설정
    net.setInput(blob)

    # 출력 레이어들에 대해 forward 수행
    outputs = net.forward(output_layers)


    # =========================
    # 10. 검출 결과 저장용 리스트
    # =========================

    # 바운딩 박스 목록
    boxes = []

    # 각 박스의 confidence 목록
    confidences = []


    # =========================
    # 11. YOLO 출력 파싱
    # =========================

    # YOLO 출력은 여러 scale의 결과로 나옴
    for output in outputs:
        # 각 detection 벡터 순회
        for detection in output:
            # 클래스별 점수 부분
            scores = detection[5:]

            # 가장 점수가 높은 클래스 번호
            class_id = np.argmax(scores)

            # 그 클래스의 confidence
            confidence = scores[class_id]

            # 차량 클래스만 남기고 confidence가 0.5보다 큰 것만 사용
            if class_id in vehicle_ids and confidence > 0.5:
                # 중심 좌표와 폭/높이를 원본 프레임 크기로 변환
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # 중심 좌표 -> 좌상단 좌표로 변환
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                # 이미지 경계를 넘지 않도록 보정
                x1 = max(0, x)
                y1 = max(0, y)
                x2 = min(width - 1, x + w)
                y2 = min(height - 1, y + h)

                # 박스 저장
                boxes.append([x1, y1, x2, y2])

                # confidence 저장
                confidences.append(float(confidence))


    # =========================
    # 12. NMS(중복 박스 제거) 적용
    # =========================

    # SORT에 넣을 최종 detection 결과
    detections = []

    # 검출된 박스가 하나라도 있으면 NMS 수행
    if len(boxes) > 0:
        # OpenCV NMSBoxes는 [x, y, w, h] 형식을 요구하므로 변환
        boxes_xywh = []
        for b in boxes:
            x1, y1, x2, y2 = b
            boxes_xywh.append([x1, y1, x2 - x1, y2 - y1])

        # confidence 0.5 이상 박스들에 대해
        # IoU 0.4 기준으로 중복 제거
        indices = cv2.dnn.NMSBoxes(boxes_xywh, confidences, 0.5, 0.4)

        # 살아남은 박스만 detections에 추가
        if len(indices) > 0:
            for i in indices.flatten():
                x1, y1, x2, y2 = boxes[i]
                conf = confidences[i]
                detections.append([x1, y1, x2, y2, conf])

    # detections를 NumPy 배열로 변환
    # detection이 하나도 없으면 빈 배열 생성
    detections = np.array(detections) if len(detections) > 0 else np.empty((0, 5))


    # =========================
    # 13. SORT 추적기 업데이트
    # =========================

    # 현재 프레임의 detection을 추적기에 넣어서
    # ID가 붙은 추적 결과를 얻음
    tracked_objects = tracker.update(detections)


    # =========================
    # 14. 추적 결과 시각화
    # =========================

    # 각 객체의 바운딩 박스와 ID를 화면에 그림
    for obj in tracked_objects:
        x1, y1, x2, y2, obj_id = obj

        # 초록색 사각형 그리기
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 사각형 위쪽에 ID 표시
        cv2.putText(
            frame,
            f"ID: {obj_id}",
            (x1, max(y1 - 10, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )


    # =========================
    # 15. 결과 영상 출력
    # =========================

    # 현재 프레임을 화면에 표시
    cv2.imshow("YOLOv3 + SORT Tracking", frame)


    # =========================
    # 16. ESC 키 입력 시 종료
    # =========================

    # 30ms 대기 후 키 입력 확인
    key = cv2.waitKey(30)

    # ESC 키(27번)가 눌리면 반복 종료
    if key == 27:
        break


# =========================
# 17. 자원 해제
# =========================

# 비디오 파일 닫기
cap.release()

# OpenCV 창 모두 닫기
cv2.destroyAllWindows()