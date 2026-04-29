# MVFS Teacher Blur-Condition v0

This package implements the first MVFS step:

1. Read DFLJPG aligned crops.
2. Build APPLE-style blur condition images: landmark ROI -> 8x8 -> upsample -> draw landmarks.
3. Create a `teacher_index.jsonl`.
4. Load teacher dataset.
5. Train a lightweight SD-Turbo teacher with PoseGuider + IDAdapter.
6. Provide simple diffusion/reconstruction/identity/temporal losses.

## Directory expectation

Recommended workspace layout:

```text
mvfs/
├─ workspace/
│  └─ person_x/
│     ├─ source/
│     │  └─ aligned_dfljpg/
│     ├─ target/
│     │  └─ aligned_dfljpg/       # student/inference later; not used by teacher blur condition training
│     └─ teacher/
│        ├─ conditions/blur_landmark/
│        └─ teacher_index.jsonl
├─ mvfs_common/
├─ datasets/
├─ models/
├─ losses/
├─ scripts/
└─ train/
```

## Build teacher conditions

```bash
python scripts/build_teacher_blur_conditions.py \
  --aligned-dir workspace/person_x/source/aligned_dfljpg \
  --out-dir workspace/person_x/teacher \
  --downsample 8 \
  --expand 0.18 \
  --debug
```

Output:

```text
workspace/person_x/teacher/conditions/blur_landmark/*.png
workspace/person_x/teacher/teacher_index.jsonl
```

## Train teacher

```bash
python train/train_teacher.py \
  --index workspace/person_x/teacher/teacher_index.jsonl \
  --output checkpoints/teacher/person_x \
  --pretrained stabilityai/sd-turbo \
  --device cuda \
  --fp16 \
  --batch-size 1 \
  --lambda-noise 1.0 \
  --lambda-recon 0.1 \
  --fixed-high-timestep 999
```

## Notes

- The teacher uses blur condition images only.
- The student should use original target aligned crops later.
- `IDAdapter` expects precomputed face-ID embeddings. The training script currently falls back to zero embeddings so that the pipeline can be smoke-tested first. Replace this with ArcFace/AdaFace/InsightFace embeddings when ready.
- The UNet is frozen by default. For RTX 3060 12GB, train PoseGuider/IDAdapter/LoRA first rather than full UNet.
