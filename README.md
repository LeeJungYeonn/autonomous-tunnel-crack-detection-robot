# Tunnel Inspection Robot for Crack Detection and Risk Mapping

> 자율 주행 로봇 기반 터널 균열 탐지 및 위험도 지도화 시스템

---

### Team  
  제로 JERO

### Member  
  2391019 이정연 · 2391022 정선영

### Project Description  
  자율 주행 로봇이 터널 중앙을 주행하며 RGB-D 카메라로 주변 벽면의 균열을 실시간으로 인식합니다.  
  이후 실제 균열 크기를 산출해 위험도를 판별하고, 균열 위치와 위험 등급을 지도 위에 시각화하는 터널 안전 진단 시스템입니다.

### Role Assignment
  **이정연 🐱 / 정선영 🐵**

  **1. 가상 환경 구성**

  - Blender 3D 터널 메쉬 모델링 및 전개도 이미지 저장 🐵
  - 균열 이미지 데이터셋 전처리 및 터널 텍스처 생성 🐱
  - Gazebo 가상 환경 구성 🐵

  **2. 로봇 및 센서 세팅**

  - 카메라, LiDAR 세팅 🐱
  - 로봇-센서 간 Transformation 계산 및 퍼블리싱 구현 🐵

  **3. 주행**

  - 좌우 벽면 오차 추출 로직 구현 🐵
  - Wall Following PID 제어기 🐵

  **4. 균열 탐지**

  - 균열 탐지 YOLO 모델 선정 및 최적화 🐱
  - 카메라 토픽 동기화 🐱

  **5. 계산**

  - 2D 픽셀 ↔ Depth 매핑 🐱
  - 3D 좌표로 BBox 실제 크기 도출 🐵

  **6. 최종 진단 결과 도출**

  - 산출된 균열 크기에 따라 `안전` / `주의` / `위험` 상태 값과 매칭되는 컬러 코드 반환 로직 구현 🐵
  - 균열을 World 좌표 및 전개도 이미지 상의 2D 좌표로 변환 🐱
  - 터널 전개도 이미지 상에 균열 마커 생성 🐵

### AI Assistance Note
  아이디어 및 전체 프로젝트 기획안을 직접 구상한 뒤, 전반적인 코드 구현 과정에서 Codex와 ChatGPT를 활용했습니다.

### References

  - **Blender 터널 모델링**  
    <https://youtu.be/vCder5cw0Ak?si=05Mf2PctH1hxgzBh>

  - **PID 기반 자율 주행**  
    <https://wikidocs.net/300880>

  - **Dataset**  
    <https://docs.ultralytics.com/datasets/segment/crack-seg>

  - **Model**  
    <https://huggingface.co/OpenSistemas/YOLOv8-crack-seg/tree/main/yolov8n/weights>

### YouTube
