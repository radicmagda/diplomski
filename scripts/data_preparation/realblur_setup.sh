# Create dataset folder
mkdir -p RealBlur_dataset

# Use gdown with the permanent file ID from your Drive link
gdown 'https://drive.google.com/uc?id=17v5Dj0M2WExgxJgsGpEWc-6UyhK1IWWK' -O RealBlur_dataset/RealBlur.tar.gz

# Extract the tar.gz file
tar -xzf RealBlur_dataset/RealBlur.tar.gz -C RealBlur_dataset

# Remove the tar.gz file after extraction
rm RealBlur_dataset/RealBlur.tar.gz

rm -r RealBlur_dataset/RealBlur-R_BM3D_ECC_IMCORR_centroid_itensity_ref
rm RealBlur_dataset/RealBlur_R_test_list.txt
rm RealBlur_dataset/RealBlur_R_train_list.txt

# Base path where your dataset lives
BASE_DIR="RealBlur_dataset"
DATASET_DIR="$BASE_DIR/RealBlur-J_ECC_IMCORR_centroid_itensity_ref"

# Destination base path
DEST_BASE="./datasets/RealBlur"

# Create required folder structure
for split in train test; do
    mkdir -p "$DEST_BASE/$split/input"
    mkdir -p "$DEST_BASE/$split/target"
done

# Function to process a given list file
process_split() {
    SPLIT_NAME=$1
    LIST_FILE="$BASE_DIR/RealBlur_J_${SPLIT_NAME}_list.txt"

    if [[ ! -f "$LIST_FILE" ]]; then
        echo "File not found: $LIST_FILE"
        return
    fi

    while read -r gt_path blur_path; do
        # Extract scene and index
        # e.g., gt_path = RealBlur-J_ECC_IMCORR_centroid_itensity_ref/scene230/gt/gt_7.png
        scene=$(basename "$(dirname "$(dirname "$gt_path")")")     # scene230
        index=$(basename "$gt_path" | grep -o '[0-9]\+')

        new_filename="${scene}_${index}.png"

        cp "$BASE_DIR/$blur_path" "$DEST_BASE/$SPLIT_NAME/input/$new_filename"
        cp "$BASE_DIR/$gt_path" "$DEST_BASE/$SPLIT_NAME/target/$new_filename"
    done < "$LIST_FILE"
}

# Process both train and test splits
process_split "train"
process_split "test"

echo "✅ Dataset restructuring complete."

rm -r RealBlur_dataset

# ========== Create Crops for Training ==========
python scripts/data_preparation/make_crops.py RealBlur
rm -rf datasets/RealBlur/train/input   # optional, remove "whole" training images
rm -rf datasets/RealBlur/train/target  # optional, remove "whole" training images
