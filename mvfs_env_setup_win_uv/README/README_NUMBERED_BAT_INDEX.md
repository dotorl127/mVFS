# Numbered 3DDFA-V3 BAT index

Helper BAT files are not numbered because they are not meant to be executed directly.

```text
bat/
├─ _3ddfa_paths.bat
├─ _3ddfa_activate.bat
├─ _3ddfa_echo_config.bat
├─ 00_3ddfa_print_config.bat
├─ 01_3ddfa_setup_env.bat
├─ 02_3ddfa_download_assets.bat
├─ 10_3ddfa_extract_video.bat
├─ 11_3ddfa_extract_source_images.bat
├─ 12_3ddfa_manual_frames.bat
└─ 13_3ddfa_debug_draw.bat
```

## Recommended order

```bat
bat\00_3ddfa_print_config.bat
bat\01_3ddfa_setup_env.bat
bat\02_3ddfa_download_assets.bat
bat\10_3ddfa_extract_video.bat person_x target target.mp4
bat\11_3ddfa_extract_source_images.bat person_x
bat\12_3ddfa_manual_frames.bat person_x target
bat\13_3ddfa_debug_draw.bat person_x target
```
