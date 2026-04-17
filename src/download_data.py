import os
from huggingface_hub import snapshot_download

def download_camus_dataset(data_dir="./data/camus"):
    print("="*50)
    print(f" STARTING DOWNLOAD DATASET CAMUS ")
    print("="*50)
    
    # Crea la directory principale se non esiste
    os.makedirs(data_dir, exist_ok=True)
    print(f"-> Folder of destination verified: {data_dir}")
    
    try:
        repo_id = "YongchengYAO/CAMUS-Lite"
        
        print(f"-> Connecting to repository: {repo_id}...")
        print("-> Starting download. This operation requires some minutes (about 3.39 GB)...")
        
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=data_dir,
            local_dir_use_symlinks=False  
        )
        
        print("\n" + "="*50)
        print(" ✅ DOWNLOAD COMPLETED! ")
        print("="*50)
        print(f"Your files are located in: {data_dir}")
        
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR during download: {e}")
        print("Suggestion: Check your internet connection and available disk space.")


import zipfile
import glob

def extract_and_cleanup(data_dir="./data/camus"):
    print("\n" + "-"*30)
    print("📦 STARTING DATASET EXTRACTION")
    print("-"*30)
    
    # Trova tutti i file zip nella cartella
    zip_files = glob.glob(os.path.join(data_dir, "*.zip"))
    
    if not zip_files:
        print("-> No .zip files found. Perhaps they have already been extracted?")
        return

    for zip_path in zip_files:
        print(f"-> Extracting: {os.path.basename(zip_path)}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        
        os.remove(zip_path)
        print(f"   ✅ Extracted and removed: {os.path.basename(zip_path)}")

    print("\n🚀 Dataset ready for use!")

if __name__ == "__main__":
    download_camus_dataset()
    extract_and_cleanup()