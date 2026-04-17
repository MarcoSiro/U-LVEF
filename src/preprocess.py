import os
import glob
import SimpleITK as sitk
import numpy as np

def preprocess_camus(input_dir="./data/camus", output_dir="./data/camus_2d"):
    print(f"Starting pre-processing: {input_dir} -> {output_dir}")
    
    out_images_dir = os.path.join(output_dir, "Images")
    out_masks_dir = os.path.join(output_dir, "Masks")
    os.makedirs(out_images_dir, exist_ok=True)
    os.makedirs(out_masks_dir, exist_ok=True)
    
    search_path = os.path.join(input_dir, "Images", "*_half_sequence.nii.gz")
    image_paths = sorted(glob.glob(search_path))
    
    if not image_paths:
        print("No files found. Please check the paths.")
        return

    processed_count = 0
    for img_path in image_paths:
        basename = os.path.basename(img_path)
        mask_name = basename.replace(".nii.gz", "_gt.nii.gz")
        mask_path = os.path.join(input_dir, "Masks", mask_name)
        
        if not os.path.exists(mask_path):
            continue
            
        # Lettura volumi 3D
        img_arr = sitk.GetArrayFromImage(sitk.ReadImage(img_path))
        mask_arr = sitk.GetArrayFromImage(sitk.ReadImage(mask_path))
        
        base_id = basename.replace("_half_sequence.nii.gz", "")
        
        np.save(os.path.join(out_images_dir, f"{base_id}_ED.npy"), img_arr[0])
        np.save(os.path.join(out_masks_dir, f"{base_id}_ED.npy"), mask_arr[0])
        
        np.save(os.path.join(out_images_dir, f"{base_id}_ES.npy"), img_arr[-1])
        np.save(os.path.join(out_masks_dir, f"{base_id}_ES.npy"), mask_arr[-1])
        
        processed_count += 2
        
    print(f"Pre-processing completed! Generated {processed_count} 2D .npy files.")

if __name__ == "__main__":
    preprocess_camus()