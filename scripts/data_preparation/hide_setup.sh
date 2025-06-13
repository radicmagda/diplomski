# ========== HIDE Dataset Setup Script ==========
ZIP_URL="https://www.dropbox.com/scl/fi/rvqdsxakk1t6qagnq1arh/HIDE_dataset.zip?rlkey=rwm7gdizn35sq8bv2dh1ja902&e=1&st=fdn5j966&dl=1"
ZIP_NAME="HIDE_dataset.zip"
EXTRACT_DIR="HIDE_dataset"
FINAL_DIR="HIDE_dataset/HIDE_dataset"
DEST_DIR="./datasets/HIDE"

# ========== Download & Extract ==========
curl -L -o "$ZIP_NAME" "$ZIP_URL"

unzip -q "$ZIP_NAME" -d "$EXTRACT_DIR"
rm "$ZIP_NAME"

# ========== Create new structure ==========
mkdir -p "$DEST_DIR/train/input" "$DEST_DIR/train/target"
mkdir -p "$DEST_DIR/test/input" "$DEST_DIR/test/target"

# ========== Move Training Images ==========
for blur_path in "$FINAL_DIR/train/"*.png; do
    filename=$(basename "$blur_path")
    cp "$blur_path" "$DEST_DIR/train/input/$filename"
    cp "$FINAL_DIR/GT/$filename" "$DEST_DIR/train/target/$filename"
done

# ========== Move Test Images ==========
for blur_path in "$FINAL_DIR/test/test-close-ups/"*.png "$FINAL_DIR/test/test-long-shot/"*.png; do
    filename=$(basename "$blur_path")
    cp "$blur_path" "$DEST_DIR/test/input/$filename"
    cp "$FINAL_DIR/GT/$filename" "$DEST_DIR/test/target/$filename"
done

# ========== Clean up old structure ==========
rm -rf "$FINAL_DIR/GT"
rm -rf "$FINAL_DIR/train"
rm -rf "$FINAL_DIR/test/test-close-ups"
rm -rf "$FINAL_DIR/test/test-long-shot"

mkdir -p ./datasets/HIDE_annotations
mv ./HIDE_dataset/HIDE_dataset/* ./datasets/HIDE_annotations/
rm -r ./HIDE_dataset

# ========== Create Crops for Training ==========
python scripts/data_preparation/make_crops.py --dataset HIDE
#rm -rf datasets/HIDE/train/input   # optional, remove "whole" training images
#rm -rf datasets/HIDE/train/target  # optional, remove "whole" training images