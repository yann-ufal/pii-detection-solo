#!/bin/bash

# The provided LUMI container ships transformers, but a version too old to include openai/privacy-filter
# This script extends the container
#
# opens a shell inside the container,
# creates a Python venv that inherits the container's site-packages (--system-site-packages)
# activates it and `pip install --upgrade transformers`
#
# Run this once on a login node before submitting the job: bash venv_and_squash.sh
# It produces ./user-software.sqsh, which run_it.sh mounts read-only into the container

set -euo pipefail

SIF=/appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260513_121430/lumi-multitorch-full-u24r70f21m50t210-20260513_121430.sif

BUILD_DIR="$PWD/build"
SQUASHFS_NAME="$PWD/user-software.sqsh"

# Optional Hugging Face token
#
#   export HF_TOKEN=hf_xxx   (before running this script)
#
# If unset, the download proceeds anonymously (per-IP rate limit)
if [ -n "${HF_TOKEN:-}" ]; then
    export SINGULARITYENV_HF_TOKEN="$HF_TOKEN"
    echo "==> Using HF_TOKEN for authenticated downloads"
else
    echo "==> HF_TOKEN not set - downloading anonymously"
fi

rm -rf "$BUILD_DIR" "$SQUASHFS_NAME"
mkdir -p "$BUILD_DIR/user-software"

echo "==> Building venv inside the container ..."
singularity exec \
    --bind "$BUILD_DIR/user-software:/user-software" \
    "$SIF" \
    bash -c '
        set -euo pipefail
        # create the venv directly at /user-software so its launcher scripts end up in /user-software/bin (matching PREPEND_PATH in run_it.sh)
        python -m venv --system-site-packages /user-software
        source /user-software/bin/activate
        VENV_SITE=$(python -c "import sysconfig; print(sysconfig.get_path('"'"'purelib'"'"'))")
        export PYTHONPATH="$VENV_SITE${PYTHONPATH:+:$PYTHONPATH}"
        python -m pip install --upgrade pip
        python -m pip install --upgrade "transformers>=5.6.0"
        echo "transformers version installed (expect >=5.6.0):"
        python -c "import transformers; print(transformers.__version__)"

        # pre-download the model weights now (login node has internet) into a cache that lives inside /user-software
        # gets baked into the .sqsh and is available offline at run time. run_it.sh sets HF_HOME=/user-software/hf
        export HF_HOME=/user-software/hf
        python -c "
from transformers import AutoModelForTokenClassification, AutoTokenizer
m = \"openai/privacy-filter\"
AutoTokenizer.from_pretrained(m)
AutoModelForTokenClassification.from_pretrained(m)
print(\"model cached under\", \"/user-software/hf\")
"
    '

echo "==> Packing venv into SquashFS: $SQUASHFS_NAME"
mksquashfs "$BUILD_DIR" "$SQUASHFS_NAME" -processors 1 -no-xattrs

echo "==> Removing build tree: $BUILD_DIR"
rm -rf "$BUILD_DIR"

echo "==> Done. Created $SQUASHFS_NAME"
