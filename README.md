# ClearML Pipeline for YOLOv9 Training and Testing

This repository contains a complete pipeline for automating YOLOv9 training and testing using ClearML. The pipeline handles dataset versioning, model training, evaluation, and deployment.

## Features

- **Dataset Versioning**: Automatically version and track your datasets
- **Model Training**: Train YOLOv9 models with configurable parameters
- **Model Evaluation**: Evaluate trained models on test data
- **Model Deployment**: Deploy models that meet performance criteria
- **Pipeline Automation**: Run the entire workflow with a single command

## Prerequisites

1. Python 3.8+
2. ClearML account (sign up at [https://app.clear.ml](https://app.clear.ml))
3. ClearML installed (`pip install clearml`)
4. ClearML configured (`clearml-init`)
5. YOLOv9 repository (included in this project)

## Project Structure

```
├── clearml_pipeline.py     # Main pipeline script
├── dataset_versioning.py   # Dataset versioning script
├── yolov9/                 # YOLOv9 repository
│   ├── data/               # Data directory
│   │   ├── dataset/        # Dataset directory
│   │   │   ├── train/      # Training data
│   │   │   ├── valid/      # Validation data
│   │   │   └── test/       # Test data
│   │   └── dataset.yaml    # Dataset configuration
│   ├── train.py            # YOLOv9 training script
│   ├── val.py              # YOLOv9 validation script
│   └── ...                 # Other YOLOv9 files
└── yolov9_training.ipynb   # Jupyter notebook for training
```

## Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/clearml-yolov9-pipeline.git
   cd clearml-yolov9-pipeline
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure ClearML:
   ```bash
   clearml-init
   ```

4. Prepare your dataset in the YOLOv9 format:
   - Place your dataset in `yolov9/data/dataset/`
   - Ensure you have the proper directory structure (train/valid/test)
   - Create a `dataset.yaml` file with your dataset configuration

## Running the Pipeline

To run the complete pipeline:

```bash
python clearml_pipeline.py
```

This will:
1. Version your dataset and upload it to ClearML
2. Train a YOLOv9 model using the dataset
3. Evaluate the trained model
4. Deploy the model if it meets the performance criteria

## Customizing the Pipeline

You can customize the pipeline by modifying the parameters in `clearml_pipeline.py`:

```python
pipeline = create_yolov9_pipeline(
    project_name="YOLOv9-Pipeline",
    pipeline_name="YOLOv9-Training-Pipeline",
    dataset_name="YOLOv9-Dataset",
    dataset_project="YOLOv9-Datasets",
    dataset_path="./yolov9/data/dataset",
    base_task_project="YOLOv9-Tasks",
)
```

## Pipeline Steps

### 1. Dataset Versioning

This step creates a versioned dataset in ClearML, which allows you to track changes to your dataset over time.

### 2. Model Training

This step trains a YOLOv9 model using the versioned dataset. You can configure:
- Number of epochs
- Batch size
- Image size
- Pre-trained weights
- Device (GPU/CPU)

### 3. Model Evaluation

This step evaluates the trained model on the test dataset and calculates metrics such as mAP, precision, and recall.

### 4. Model Deployment

This step deploys the model if it meets the specified performance criteria (e.g., mAP > 0.5).

## Monitoring

You can monitor the pipeline and all its steps in the ClearML web UI at [https://app.clear.ml](https://app.clear.ml).

## Advanced Usage

### Running on Remote Machines

You can run the pipeline on remote machines using ClearML Agent:

1. Install ClearML Agent on the remote machine:
   ```bash
   pip install clearml-agent
   ```

2. Configure the agent:
   ```bash
   clearml-agent init
   ```

3. Start the agent:
   ```bash
   clearml-agent daemon --queue default
   ```

4. The pipeline will automatically use the remote agent for execution.

### Hyperparameter Optimization

You can extend the pipeline to include hyperparameter optimization by adding a step that uses ClearML's HyperParameterOptimizer.

## Troubleshooting

- **Dataset not found**: Ensure your dataset path is correct and accessible
- **GPU not detected**: Check your CUDA installation and GPU drivers
- **ClearML connection issues**: Verify your ClearML credentials and network connection

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [YOLOv9](https://github.com/WongKinYiu/yolov9) for the object detection model
- [ClearML](https://clear.ml) for the MLOps platform 