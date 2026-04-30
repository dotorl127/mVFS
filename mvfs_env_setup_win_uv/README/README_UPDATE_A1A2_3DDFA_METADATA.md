# update_a1a2_3ddfa_metadata_aligned_v3

이 버전은 기존 A1/A2 DFLJPG aligned crop에 3DDFA-V3 metadata를 추가하고, 원하면 top-level `meta["landmarks"]`까지 3DDFA-V3 landmark로 교체한다.

## 현재 v2/v3 metadata 상태

v2부터는 이미 아래에 3DDFA-V3 landmark가 저장된다.

```python
meta["mvfs"]["3ddfa_v3"]["landmarks"]
```

다만 v2에서는 top-level:

```python
meta["landmarks"]
```

는 기존 DFL landmark 그대로였다.

## v3 변경점

`--replace-top-landmarks` 옵션을 주면:

```python
meta["mvfs"]["landmarks_dfl_backup"] = 기존 meta["landmarks"]
meta["landmarks"] = meta["mvfs"]["3ddfa_v3"]["landmarks"]
meta["mvfs"]["top_landmarks_source"] = "3ddfa_v3"
```

즉 DFL을 더 이상 쓰지 않는 MVFS 전용 데이터셋으로 변환한다.

## 실행

```bat
cd D:\MVFS\3DDFA-V3

python update_a1a2_3ddfa_metadata_aligned_v3.py ^
  --dataset-root D:\MVFS\dataset\mfvs_dataset ^
  --splits A1,A2 ^
  --device cuda ^
  --backbone resnet50 ^
  --write-dfljpg ^
  --write-npz ^
  --replace-top-landmarks
```

bat:

```bat
bat\14_3ddfa_update_a1a2_metadata.bat D:\MVFS\dataset\mfvs_dataset
```

## 확인

```bat
python check_dfljpg_mvfs_metadata.py D:\MVFS\dataset\mfvs_dataset\m_000001\A2\frame_000001.jpg
```

정상 출력 예:

```text
3ddfa_v3 landmarks: True len= 68
top_landmarks_source: 3ddfa_v3
has dfl backup: True
3dmm keys: ['id_coeff', 'exp_coeff', ...]
```

## skip-existing 주의

이미 v2로 한 번 처리한 파일은 `--skip-existing` 때문에 v3에서 top-level landmark 교체가 안 될 수 있다.

그 경우에는 아래 둘 중 하나를 선택한다.

1. `--skip-existing`를 빼고 재실행
2. 별도 migration 스크립트로 기존 `mvfs.3ddfa_v3.landmarks`를 top-level로 복사

가장 단순한 방법:

```bat
python update_a1a2_3ddfa_metadata_aligned_v3.py ^
  --dataset-root D:\MVFS\dataset\mfvs_dataset ^
  --splits A1,A2 ^
  --device cuda ^
  --backbone resnet50 ^
  --write-dfljpg ^
  --write-npz ^
  --replace-top-landmarks
```
