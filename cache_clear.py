import os
import shutil

# Path to Keras cache
cache_dir = os.path.join(os.path.expanduser('~'), '.keras', 'models')
print(f"Cache directory: {cache_dir}")

# Remove all EfficientNet cached files
if os.path.exists(cache_dir):
    for file in os.listdir(cache_dir):
        if 'efficientnet' in file.lower():
            file_path = os.path.join(cache_dir, file)
            print(f"Deleting: {file_path}")
            os.remove(file_path)
    print("✅ Cache cleared!")
else:
    print("No cache directory found")