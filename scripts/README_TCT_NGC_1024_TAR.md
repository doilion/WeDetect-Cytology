# TCT_NGC_1024

This dataset is the 1024x1024 letterboxed TCT_NGC cache prepared for WeDetect
experiments. It is packaged as tar archives to make ModelScope upload and
download faster than transferring 166k individual image files.

## Contents

- `annotations/`
  - `instances_train_dev_disjoint_dev30.json`
  - `instances_val_dev_disjoint_dev30.json`
  - `instances_test_base_clean_dev30.json`
  - `instances_test_novel_merged_9.json`
- `archives/images_Serous_effusion.tar`
- `archives/images_TCT_CCD.tar`
- `archives/images_Thyroid_gland.tar`
- `archives/images_Urine.tar`
- `archives/images_respiratory_tract.tar`
- `SHA256SUMS`

## Restore The Original Layout

Option 1: use the helper script from the WeDetect repository:

```bash
bash scripts/download_tct_ngc_1024_modelscope_tar.sh /path/to/TCT_NGC_1024
```

Option 2: run these commands after downloading the dataset snapshot:

```bash
mkdir -p images
tar -xf archives/images_Serous_effusion.tar -C images
tar -xf archives/images_TCT_CCD.tar -C images
tar -xf archives/images_Thyroid_gland.tar -C images
tar -xf archives/images_Urine.tar -C images
tar -xf archives/images_respiratory_tract.tar -C images
sha256sum -c SHA256SUMS
```

After extraction, the dataset root has the expected layout:

```text
TCT_NGC_1024/
  annotations/
  images/
    Serous_effusion/
    TCT_CCD/
    Thyroid_gland/
    Urine/
    respiratory_tract/
```

All image metadata in the annotation JSON files is 1024x1024, and the bounding
boxes were resized for this 1024 cache.
