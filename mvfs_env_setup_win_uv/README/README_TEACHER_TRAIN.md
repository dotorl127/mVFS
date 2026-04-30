# MVFS Teacher Train Debug v2

## 지금 티처 학습 가능한가?

가능하다. 전제는 아래 3개다.

1. `workspace/general_teacher/meta/teacher_index.jsonl` 존재
2. `workspace/general_teacher/conditions/blur_only/...` 존재
3. `workspace/general_teacher/id_embeddings/*.npy` 존재

즉 네가 앞에서 만든 A1/A2 -> teacher dataset 빌드가 끝났으면,
이제 바로 teacher 학습을 돌릴 수 있다.

---

## 이번 수정 내용

### 1) 디버깅 이미지 저장
학습 중 아래 순서로 패널 이미지를 저장한다.

```text
ID | BLUR CONDITION | RECON | GT
```

저장 위치:

```text
<teacher_workspace>/runs/<run_name>/debug/step_0000100.jpg
```

### 2) ID 이미지 자동 추론
`teacher_index.jsonl`에 `identity_path`가 없어도 괜찮다.

학습용 Dataset이 자동으로:

```text
clean_path = .../<person_id>/A2/xxx.jpg
```

를 보고, 같은 person의:

```text
.../<person_id>/A1/
```

에서 첫 번째 이미지를 찾아서 ID debug image로 사용한다.

즉 지금 네 데이터 구조에서도 바로 동작한다.

---

## 덮어쓸 파일

```text
datasets/teacher_blur_dataset.py
train/train_teacher.py
bat/26_teacher_train.bat
```

---

## 실행

```bat
bat\26_teacher_train.bat D:\MVFS\workspace\general_teacher teacher_run01
```

---

## 출력 구조 예시

```text
workspace/general_teacher/
└─ runs/
   └─ teacher_run01/
      ├─ train_config.json
      ├─ checkpoints/
      │  ├─ teacher_adapters_step_0001000.pt
      │  └─ ...
      └─ debug/
         ├─ step_0000000.jpg
         ├─ step_0000100.jpg
         └─ ...
```

---

## 참고

- `--debug-image-every 100`
  - 100 step마다 디버그 저장
- `--debug-num-samples 1`
  - 한 장에 몇 샘플을 넣을지
- `--debug-max-save 200`
  - 디버그 파일 최대 저장 개수
- `--recon-every 4`
  - recon decode 주기

디버그 이미지를 더 자주 보고 싶으면:

```text
--debug-image-every 20
```

처럼 줄이면 된다.
