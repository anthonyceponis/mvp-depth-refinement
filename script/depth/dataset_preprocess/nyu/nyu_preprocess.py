import h5py
import numpy as np
from PIL import Image
import os

BASE_DATA_DIR = os.environ["BASE_DATA_DIR"]

MAT_FILE_PATH = BASE_DATA_DIR + "nyu_v2_raw/nyu_depth_v2_labeled.mat"
OUTPUT_DIR = BASE_DATA_DIR + "nyu_v2_processed"
TRAIN_LIST_PATH = "data_split/nyu_depth/labeled/filename_list_train.txt"
TEST_LIST_PATH = "data_split/nyu_depth/labeled/filename_list_test.txt"


# --- Helper function to create directories ---
def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


# --- Main processing logic ---
def process_nyu_dataset():
    print(f"Loading .mat file using h5py from: {MAT_FILE_PATH}")
    try:
        with h5py.File(MAT_FILE_PATH, "r") as f:
            # Get dataset objects without loading them into memory
            images_ds = f["images"]
            depths_ds = f["depths"]
            rawDepths_ds = f["rawDepths"]

            # The first dimension is the number of samples (N)
            N = images_ds.shape[0]
            print(f"Found {N} samples in the .mat file.")

            # Load filename lists
            all_filenames = []
            for list_path in [TRAIN_LIST_PATH, TEST_LIST_PATH]:
                with open(list_path, "r") as f_list:
                    for line in f_list:
                        all_filenames.append(line.strip().split())

            if len(all_filenames) != N:
                print(
                    f"Warning: Mismatch between number of samples in .mat file ({N}) and combined filename lists ({len(all_filenames)})."
                )
                print(
                    "Proceeding, but this might indicate an issue with the dataset split files or the .mat file itself."
                )

            print(f"Processing {N} samples...")
            for i in range(N):
                if i % 100 == 0:
                    print(f"  Processing sample {i+1}/{N}")

                # Get target paths from filename list
                try:
                    rgb_rel_path, depth_rel_path, filled_rel_path = all_filenames[i]
                except IndexError:
                    print(
                        f"Error: Could not find filename entry for sample {i}. Skipping."
                    )
                    continue

                # --- Process RGB image ---
                # Slice one image (C, W, H) and transpose to (H, W, C) for PIL
                rgb_img_data = images_ds[i, :, :, :].transpose(2, 1, 0)
                rgb_image = Image.fromarray(rgb_img_data)
                rgb_output_path = os.path.join(OUTPUT_DIR, rgb_rel_path)
                ensure_dir(rgb_output_path)
                rgb_image.save(rgb_output_path)

                # --- Process Raw Depth image ---
                # Slice one image (W, H) and transpose to (H, W)
                raw_depth_data = rawDepths_ds[i, :, :].transpose(1, 0)
                # Convert meters to millimeters and save as 16-bit PNG
                raw_depth_mm = (raw_depth_data * 1000).astype(np.uint16)
                raw_depth_image = Image.fromarray(raw_depth_mm)
                raw_depth_output_path = os.path.join(OUTPUT_DIR, depth_rel_path)
                ensure_dir(raw_depth_output_path)
                raw_depth_image.save(raw_depth_output_path)

                # --- Process Filled Depth image ---
                # Slice one image (W, H) and transpose to (H, W)
                filled_depth_data = depths_ds[i, :, :].transpose(1, 0)
                # Convert meters to millimeters and save as 16-bit PNG
                filled_depth_mm = (filled_depth_data * 1000).astype(np.uint16)
                filled_depth_image = Image.fromarray(filled_depth_mm)
                filled_depth_output_path = os.path.join(OUTPUT_DIR, filled_rel_path)
                ensure_dir(filled_depth_output_path)
                filled_depth_image.save(filled_depth_output_path)

    except Exception as e:
        print(f"An error occurred during processing: {e}")
        import traceback

        traceback.print_exc()
        return

    print(f"Dataset processing complete. Images saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    process_nyu_dataset()
