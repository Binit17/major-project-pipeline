import os
import shutil
from pathlib import Path

def consolidate_dataset(dataset_path="Dataset", 
                         final_nonstrike_folder="final_nonstrike", 
                         final_strike_folder="final_strike"):
    """
    Consolidate all images from multiple nonstrike/strike folders into two final folders.
    
    Args:
        dataset_path: Path to the dataset directory
        final_nonstrike_folder: Name of the folder to store all nonstrike images
        final_strike_folder: Name of the folder to store all strike images
    """
    # Create final destination folders if they don't exist
    Path(dataset_path, final_nonstrike_folder).mkdir(exist_ok=True)
    Path(dataset_path, final_strike_folder).mkdir(exist_ok=True)
    
    # Get all subdirectories in the dataset folder
    subdirs = [d for d in os.listdir(dataset_path) 
               if os.path.isdir(os.path.join(dataset_path, d))]
    
    # Counter for stats
    nonstrike_count = 0
    strike_count = 0
    
    # Process each subdirectory
    for subdir in subdirs:
        # Skip the final folders themselves
        if subdir in [final_nonstrike_folder, final_strike_folder]:
            continue
        
        src_dir = os.path.join(dataset_path, subdir)
        
        # Determine if this is a strike or nonstrike folder
        if subdir.startswith('nonstrike'):
            dest_dir = os.path.join(dataset_path, final_nonstrike_folder)
            for file in os.listdir(src_dir):
                # Check if it's an image file (you can add more extensions if needed)
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    src_file = os.path.join(src_dir, file)
                    # Create a unique filename to avoid overwriting
                    base, ext = os.path.splitext(file)
                    dest_file = os.path.join(dest_dir, f"{subdir}_{file}")
                    
                    # Copy the file
                    shutil.copy2(src_file, dest_file)
                    nonstrike_count += 1
                    
        elif subdir.startswith('strike'):
            dest_dir = os.path.join(dataset_path, final_strike_folder)
            for file in os.listdir(src_dir):
                # Check if it's an image file
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    src_file = os.path.join(src_dir, file)
                    # Create a unique filename to avoid overwriting
                    base, ext = os.path.splitext(file)
                    dest_file = os.path.join(dest_dir, f"{subdir}_{file}")
                    
                    # Copy the file
                    shutil.copy2(src_file, dest_file)
                    strike_count += 1
    
    print(f"Consolidation complete!")
    print(f"Copied {nonstrike_count} images to {final_nonstrike_folder}")
    print(f"Copied {strike_count} images to {final_strike_folder}")
    
if __name__ == "__main__":
    consolidate_dataset()