# MVFS Teacher Late ID Loss FaceNet v1

## 결론

RTX 3060 12GB 기준 ID loss 모델은 일단 **facenet-pytorch InceptionResnetV1 pretrained=vggface2**로 선택했다.

선택 이유:

```text
- PyTorch native라 recon -> ID loss -> UNet gradient가 흐름
- ArcFace iresnet100 / Arc2Face / PulID보다 가벼움
- RTX 3060 12GB에서 full UNet 학습과 같이 쓰기 현실적
- 나중에 TorchScript ArcFace로 교체 가능
```

## 학습 정책

사용자 의도대로 **초반에는 ID loss를 쓰지 않고 후반부에만 사용**한다.

기본값:

```text
max_steps          = 65000 optimizer steps
id_loss_start_step = 50000
```

즉:

```text
0 ~ 49999 step:
  noise + 10 * recon

50000 ~ 65000 step:
  noise + 10 * recon + 1 * id
```

SD-Turbo 기반이라 timestep filtering은 넣지 않았다.

## 설치

```bat
bat\04_mvfs_install_idloss_facenet.bat
```

## 실행

```bat
bat\26_teacher_train_late_idloss_facenet.bat D:\MVFS\workspace\general_teacher late_idloss_run01
```

## 기본 학습 설정

```text
--train-unet
--fp16
--gradient-checkpointing
--batch-size 1
--grad-accum-steps 8
--lr 1e-4
--weight-decay 1e-3
--lambda-noise 1.0
--lambda-recon 10.0
--lambda-id 1.0
--id-loss-start-step 50000
```

## TorchScript ArcFace로 바꾸고 싶을 때

`facenet` 대신 직접 준비한 TorchScript 모델을 쓸 수 있다.

```bat
python train\train_teacher.py ^
  ... ^
  --id-loss-backend torchscript ^
  --id-loss-model D:\MVFS\pretrained\id\arcface_jit.pt ^
  --id-loss-input-range minus1_1
```

## 주의

기존 InsightFace ONNX buffalo_l은 계속 쓴다.

```text
ONNX buffalo_l:
  A1 평균 ID embedding 생성
  ID conditioning token용

FaceNet/TorchScript:
  recon image에 대한 differentiable ID loss용
```
