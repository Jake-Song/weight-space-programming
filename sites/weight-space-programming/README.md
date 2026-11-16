# weight-space-programming — 로컬 실험 비교 웹페이지

외부 배포 없이 사용하는 정적 HTML/CSS/JavaScript 페이지입니다.

저장소 루트에서 실행:

```sh
python3 -m http.server 4173 --bind 127.0.0.1 --directory sites/weight-space-programming/dist
```

브라우저에서 http://localhost:4173 을 엽니다. 종료는 실행 터미널에서 Ctrl+C입니다.
첫 화면에 3D 모델, 바로 아래에 정확도·비용 비교를 표시합니다. 두 섹션을 한 페이지에 통합했습니다.

현재 작업 환경에서는 세션 종료에 영향받지 않도록 `weight-space-programming.service` 사용자 서비스로 실행합니다.
이 서비스는 재부팅 시 자동 시작하도록 등록하지 않은 임시 서비스입니다.

```sh
systemctl --user status weight-space-programming.service
systemctl --user stop weight-space-programming.service
```

- 24개 레이어의 3D 모델과 수정된 15·19번 레이어 탐색
- 6개 조건의 정답률·토큰 한도 도달률·평균 생성 토큰 차트
- 조건 선택 및 차이·탐색적 95% 구간 표시
- 전체 수치 표와 결과 JSON 다운로드

`dist/results.json`은 저장소의 `results/summary.json` 사본입니다.
화면의 6개 조건 수치를 원본과 대조했습니다.
Google Fonts 연결이 없으면 시스템 글꼴을 사용합니다.

## JavaScript 3D 모델

http://localhost:4173 에서 회전·확대하고 레이어를 선택할 수 있습니다.
이전 `model.html` 주소는 같은 통합 페이지로 이동합니다.
모든 3D 형태와 레이블을 브라우저에서 JavaScript로 생성합니다.
Three.js 0.180.0과 필요한 모듈·MIT 라이선스를 로컬에 포함합니다.

- `dist/model-geometry.js`: 24개 레이어, attention·MLP, 수정된 열 64개씩, 카메라를 바라보는 레이블 생성
- `dist/model.js`: 회전·줌·레이어 선택·간격 조절·초기 위치 복원
- `dist/assets/model-config.json`: 실험과 동일한 모델 revision의 설정
- `dist/assets/model-metadata.json`: 구조와 수정 명세

입력에서 출력까지 왼쪽에서 오른쪽으로 배치했습니다.
15번은 보라색, 19번은 민트색이며 인덱스는 0부터 시작합니다.
확대·축소는 고정된 중심을 기준으로 작동합니다. 두 손가락 확대는 레이어 선택으로 처리하지 않습니다.
`처음 위치로`를 누르면 회전·확대·간격·표시 조건을 초기 상태로 되돌립니다.
JavaScript 모듈과 명세 로딩을 위해 위 HTTP 서버를 사용하세요.

모델은 텍스트 백본의 구조 모식도입니다. 비전 인코더·세부 정규화·잔차 연산은 생략했습니다.
64개 발광 블록은 수정된 열 64개를 나타내며 실제 열 인덱스 위치를 재현하지 않습니다.
간격·크기·밝기는 실제 파라미터 크기나 활성값을 나타내지 않습니다.

이전 Blender 산출물은 `archive/blender/`에 보관하며 웹페이지에서는 사용하지 않습니다.
