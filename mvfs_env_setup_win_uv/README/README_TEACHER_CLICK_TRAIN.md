# MVFS Teacher Blur Reconstruction "click train" add-on

이 패키지는 teacher blur condition 복원 학습을 한 번에 실행하기 위한 추가 파일이다.

## teacher_index.jsonl의 역할

`teacher_index.jsonl`은 **학습 샘플 목록 파일**이다.  
이미지 파일을 폴더에서 매번 다시 스캔하지 않고, 각 샘플에 필요한 경로와 메타데이터를 한 줄씩 기록한다.

한 줄은 보통 이런 형태다.

```json
{
  "clean_path": "workspace/person_x/source/aligned_dfljpg/img_0001_00.jpg",
  "condition_path": "workspace/person_x/teacher_source/conditions/blur_landmark/img_0001_00.png",
  "source_filename": "img_0001_00.jpg",
  "landmarks": "... optional ...",
  "id_embed_path": "workspace/person_x/teacher_source/id_embeddings/img_0001_00.npy",
  "3dmm_path": "workspace/person_x/teacher_source/3dmm/img_0001_00.npz"
}
```

역할:

```text
clean_path       : 복원 정답 aligned crop
condition_path   : 8x8 down/up blur + landmark overlay condition image
id_embed_path    : InsightFace buffalo_l identity embedding
3dmm_path        : DFLJPG에서 추출한 3DMM coeff 캐시, 있으면 사용
```

즉 dataset loader는 `teacher_index.jsonl`만 읽으면 학습에 필요한 clean/condition/id/3dmm 정보를 바로 찾을 수 있다.

## 추가 파일

```text
scripts/
├─ build_teacher_id_embeddings.py
└─ attach_3dmm_to_teacher_index.py

bat/
├─ _mvfs_paths.bat
├─ _mvfs_activate.bat
├─ 20_teacher_build_blur_conditions.bat
├─ 21_teacher_build_id_embeddings.bat
├─ 22_teacher_attach_3dmm_index.bat
├─ 23_teacher_train.bat
└─ 24_teacher_all.bat
```

## 실행

전체 실행:

```bat
bat\24_teacher_all.bat person_x source
```

단계별 실행:

```bat
bat\20_teacher_build_blur_conditions.bat person_x source
bat\21_teacher_build_id_embeddings.bat person_x source
bat\22_teacher_attach_3dmm_index.bat person_x source
bat\23_teacher_train.bat person_x source
```

## 출력

```text
workspace/person_x/teacher_source/
├─ conditions/blur_landmark/
├─ id_embeddings/
├─ 3dmm/
├─ teacher_index.jsonl
├─ teacher_index_with_id.jsonl
├─ teacher_index_final.jsonl
└─ id_embedding_mean.npz

checkpoints/teacher/person_x_source/
```

## 주의

- `build_teacher_blur_conditions.py`와 `train_teacher.py`는 기존 teacher blur v0 패키지의 파일을 사용한다.
- `InsightFace buffalo_l`은 `.venv-mvfs` 환경에 설치되어 있어야 한다.
- `3dmm_path`는 DFLJPG 안에 `mvfs.3dmm` metadata가 있을 때만 생성된다. 없어도 학습은 진행 가능하다.
