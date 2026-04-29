# Refactored MVFS / 3DDFA-V3 BAT layout

이 버전은 `mvfs_3ddfa_v3_bat`의 bat들을 `mvfs_env_setup_win_uv` 스타일의 `mvfs/bat/` 아래로 모으고, 공통 경로 설정을 별도 bat로 분리한 구조다.

## 구조

```text
mvfs/
├─ bat/
│  ├─ _3ddfa_paths.bat
│  ├─ _3ddfa_activate.bat
│  ├─ _3ddfa_echo_config.bat
│  ├─ 01_3ddfa_setup_env.bat
│  ├─ 02_3ddfa_download_assets.bat
│  ├─ 10_3ddfa_extract_video.bat
│  ├─ 11_3ddfa_extract_source_images.bat
│  ├─ 12_3ddfa_manual_frames.bat
│  ├─ 13_3ddfa_debug_draw.bat
│  └─ 00_3ddfa_print_config.bat
├─ env/
│  └─ requirements-3ddfa-official-py38-cu102-win.txt
├─ tools/
│  └─ download_3ddfa_assets.py
├─ .venv-3ddfa/
├─ mvfs_common/
├─ 3ddfa-v3/
└─ workspace/
```

## 공통 경로 bat

모든 3DDFA bat는 내부적으로 아래 파일을 호출한다.

```bat
bat\_3ddfa_paths.bat
bat\_3ddfa_activate.bat
```

기본값:

```bat
MVFS_ROOT  = mvfs 루트
DDDFA_ROOT = %MVFS_ROOT%\3ddfa-v3
DDDFA_VENV = %MVFS_ROOT%\.venv-3ddfa
```

외부에서 override도 가능하다.

```bat
set DDDFA_ROOT=D:\repos\3DDFA-V3
set DDDFA_VENV=D:\envs\3ddfa
bat\00_3ddfa_print_config.bat
```

## 설치

```bat
bat\01_3ddfa_setup_env.bat
bat\02_3ddfa_download_assets.bat
```

## 실행

target video:

```bat
bat\10_3ddfa_extract_video.bat person_x target target.mp4
```

source video:

```bat
bat\10_3ddfa_extract_video.bat person_x source source.mp4
```

source image folder:

```bat
bat\11_3ddfa_extract_source_images.bat person_x
```

manual fix:

```bat
bat\12_3ddfa_manual_frames.bat person_x target
```

debug draw:

```bat
bat\13_3ddfa_debug_draw.bat person_x target
```

## 경로 규칙

video:

```text
workspace/<person>/<target|source>/raw/videos/<video_name>
```

source images:

```text
workspace/<person>/source/raw/images
```

frames for manual:

```text
workspace/<person>/<target|source>/frames
```

aligned output:

```text
workspace/<person>/<target|source>/aligned_dfljpg
```

debug:

```text
workspace/<person>/<target|source>/debug/...
```
