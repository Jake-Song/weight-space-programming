# weight-space-programming — 로컬 실험 대시보드

외부 Sites 배포 없이 사용하는 정적 HTML/CSS/JavaScript 페이지입니다.

```sh
python3 -m http.server 4173 --bind 127.0.0.1 --directory sites/weight-space-programming/dist
```

브라우저에서 http://localhost:4173 을 엽니다. 현재는 대화 종료 후에도 볼 수 있도록
`weight-space-programming.service` 사용자 서비스로 실행합니다.
재부팅 시 자동 시작하도록 등록한 서비스는 아닙니다.

```sh
systemctl --user status weight-space-programming.service
systemctl --user stop weight-space-programming.service
```

## 화면 구성

- 첫 화면: 2026-11-17 독립 10단계 진단, GPU 3회·Python 7회·가중치 편집 0개.
- 후보 층 선택: 10·19·23번 층의 4개 대조 쌍 비율, 고정 채널과 사전 판정 기준.
- 3D 모델: 기본은 이번 진단 후보. 보기 전환으로 이전 15·19번 층 공개 편집을 확인.
- 10단계 실행 기록: 오류와 부분 분석을 포함하며 각 항목을 펼쳐 상세 확인.
- 이전 실험: 2026-11-16 공개 편집 재현의 6개 조건 정확도·비용 비교.

이번 진단에서는 가중치를 수정하지 않았습니다. 10번 층의 후보 집합만 대조 기준을
충족했지만, 합성된 짧은 문맥의 관찰이며 반복의 인과 원인이나 편집 효과를
검증한 결과는 아닙니다. 이전 실험의 정확도 수치는 별도로 표시합니다.

## 데이터와 파일

- `dist/diagnostics.json`: 이번 실행의 집계·후보·단계 기록.
- `dist/reports/`: `results/runs/ten-steps-20261117.md`와 상세 지표 JSON 사본.
- 후보 비율은 `results/runs/ten-steps-20261117-metrics.json`에서 가져옵니다.
- 고정 채널과 대조 설계는 `.sandbox/work/action-000009-controls/protocol.json`에 기록돼 있습니다.
- `dist/results.json`: 이전 실험의 `results/summary.json` 사본.
- `dist/diagnostic.js`, `dist/diagnostic.css`: 최신 진단 차트·진행 기록.
- `dist/model-geometry.js`, `dist/model.js`: 브라우저에서 생성하는 3D 구조와 조작.
- `dist/assets/model-config.json`, `model-metadata.json`: 고정 revision 모델 구조와 이전 편집 명세.

Google Fonts 연결이 없으면 시스템 글꼴을 사용합니다.
Three.js와 MIT 라이선스는 로컬에 포함합니다.

## 3D 표시와 조작

24개 텍스트 레이어를 입력에서 출력까지 수평으로 배치합니다. 드래그로 회전,
휠·두 손가락으로 중심을 유지한 확대/축소, 레이어 선택, 간격 조절을 지원합니다.
`처음 위치로`는 현재 실험 보기 안에서 회전·확대·간격·필터를 초기화합니다.
`model.html`의 기존 주소도 통합 화면으로 이동합니다.

이번 진단의 발광 블록은 L10의 3개, L19의 3개, L23의 5개 고정 후보 채널입니다.
이전 공개 편집 보기에서는 L15·L19 각각 64개 수정 열을 표시합니다.
층 인덱스는 0부터 시작하며, 블록 배치는 실제 열 인덱스를 재현하지 않습니다.
색·크기·밝기도 실제 활성값을 나타내지 않습니다. 비전 인코더와 세부 잔차 연산은 생략했습니다.
이전 Blender 산출물은 `archive/blender/`에 보관하며 페이지에서 사용하지 않습니다.

## 브라우저 검증

실행 중인 로컬 서버와 Playwright/Chromium이 필요합니다.
기본 경로를 사용할 수 없으면 모듈과 실행 파일을 지정합니다.

```sh
PLAYWRIGHT_MODULE=/absolute/path/to/playwright \
PLAYWRIGHT_CHROMIUM=/absolute/path/to/chrome \
node scripts/check_model_viewer.cjs
```

원본 지표와 화면 수치, 기준 판정, 후보/편집 보기 전환, 실행 기록,
기존 차트, 회전·확대·초기화, 모바일 너비와 JavaScript 오류를 확인합니다.
