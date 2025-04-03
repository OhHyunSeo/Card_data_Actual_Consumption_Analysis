# 💳 Card Data Actual Consumption Analysis

[![Python Version](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)

## 🔍 소개

이 프로젝트는 카드 매출 데이터를 바탕으로 소비 패턴을 분석하고,  
특정 이벤트(예: 신년 행사)와의 연관성 및 소비 흐름을 파악하기 위한 텍스트 및 빈도 분석을 포함합니다.

---

## 🧩 주요 기능

- 💰 카드 사용 데이터 기반 시계열 소비 분석
- 📊 이벤트-코스 간 빈도 분석 (`event-course-name-freq.csv`)
- 🧠 텍스트 기반 이벤트 분석 (신년, 시즌 등)
- 🧪 MPS 흐름 실험 (예측 또는 수요분석 개념)

---

## 📁 프로젝트 구조

```
📁 Card_data_Actual_Consumption_Analysis-master/
│
├── MPS_test.py                      # MPS 분석 또는 소비 흐름 시뮬레이션
├── card_analysis.py                # 주요 카드 소비 분석
├── event-course-name-freq.csv      # 이벤트 및 코스 빈도 집계 결과
└── neyyear_event_text_analysis.py  # 신년 이벤트 관련 텍스트 분석
```

---

## 🚀 실행 방법

### 1. 가상환경 및 라이브러리 설치

```bash
python -m venv venv
source venv/bin/activate
pip install pandas matplotlib seaborn
```

### 2. 실행 예시

```bash
python card_analysis.py                 # 카드 데이터 분석 메인 실행
python neyyear_event_text_analysis.py   # 신년 이벤트 텍스트 분석 실행
python MPS_test.py                      # 소비 흐름 분석 시뮬레이션
```

---

## 📈 분석 예시

- 월별 소비 변화, 지역/카테고리별 소비 패턴
- 특정 시즌(예: 연말연시)에 소비 집중 여부
- 이벤트 참여 빈도와 소비 연관성 시각화

---

## 🧑‍💻 기여 방법

1. 이 프로젝트를 포크합니다.
2. 새로운 브랜치를 생성합니다: `git checkout -b feature/기능`
3. 변경사항을 커밋합니다: `git commit -m "Add 기능"`
4. 브랜치에 푸시합니다: `git push origin feature/기능`
5. Pull Request를 생성합니다.

