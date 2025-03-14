from clearml import Dataset

# 1. Create (or load) a dataset
dataset = Dataset.create(
    dataset_name='YOLOv9-Dataset',
    dataset_project='YOLOv9-Project'
)

# 2. Add your local dataset folder (recursively includes subfolders)
dataset.add_files('./yolov9/data/dataset')

# 3. Upload files to a storage destination
#    - For a local folder, use file:///...
#    - For S3, use s3://...
#    - For GCS, use gs://...
dataset.upload(output_url='./dummy')

# 4. Finalize to lock in this version
dataset.finalize()
