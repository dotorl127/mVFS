# MVFS Teacher Dataset Builder v4 (soft blur)

## 변경점

이 버전은 teacher blur condition을 `8x8 nearest mosaic` 방식이 아니라
APPLE 의도에 더 가까운 `soft blur` 방식으로 바꾼 버전이다.

```text
입력 이미지 전체를 downsample -> upsample(bicubic) -> gaussian blur
그리고 얼굴 영역 mask에만 feather blend
```

즉:

- 배경은 최대한 유지
- 얼굴 영역은 고주파 identity detail 제거
- 경계는 feathering으로 자연스럽게 연결

## landmark 사용

teacher blur condition 생성용 landmark는 기본적으로:

```text
--landmark-source 3ddfa
```

를 사용한다.

향후 student pose encoder 쪽은 DreamID처럼
`3DMM 합성 -> 2D projection landmark`를 그대로 인코더에 넣는 단순 설계로 가면 된다.
이 파일은 teacher blur dataset 생성만 담당한다.

## 실행

```bat
bat\25_teacher_build_from_a1a2.bat D:\MVFS\dataset\mfvs_dataset D:\MVFS\workspace\general_teacher
```

직접 실행:

```bat
python scripts\build_teacher_dataset_from_a1a2.py ^
  --dataset-root D:\MVFS\dataset\mfvs_dataset ^
  --out-root D:\MVFS\workspace\general_teacher ^
  --id-source A1 ^
  --id-embed-mode direct ^
  --landmark-source 3ddfa ^
  --downsample 32 ^
  --expand 0.20 ^
  --blur-sigma 2.0 ^
  --feather-sigma 8.0 ^
  --debug ^
  --fallback-zero-id
```

## 출력

```text
workspace/general_teacher/
├─ conditions/
│  ├─ blur_only/
│  └─ debug_blur_landmark/
├─ id_embeddings/
├─ meta/
│  ├─ teacher_index.jsonl
│  ├─ id_embedding_report.jsonl
│  └─ landmark_source_counts.json
└─ skipped.jsonl
```

`debug_blur_landmark`는 확인용이다.
실제 학습 입력은 `conditions/blur_only`만 사용한다.
