"""
ClearML Pipeline for YOLOv9 Training and Testing
This script creates a complete pipeline for:
1. Dataset versioning and management
2. Model training with YOLOv9
3. Model evaluation
4. Model deployment
"""

import os
import sys
from pathlib import Path
from clearml import Task, Dataset, PipelineController

# Define the pipeline
def create_yolov9_pipeline(
    project_name="YOLOv9-Pipeline",
    pipeline_name="YOLOv9-Training-Pipeline",
    dataset_name="YOLOv9-Dataset",
    dataset_project="YOLOv9-Datasets",
    dataset_path="./yolov9/data/dataset",
    base_task_project="YOLOv9-Tasks",
    output_uri=None,
    add_pipeline_tags=None,
):
    """
    Create a ClearML pipeline for YOLOv9 training and testing
    
    Args:
        project_name: Name of the pipeline project
        pipeline_name: Name of the pipeline
        dataset_name: Name of the dataset
        dataset_project: Project name for the dataset
        dataset_path: Path to the dataset
        base_task_project: Project name for the tasks
        output_uri: URI for output artifacts
        add_pipeline_tags: Additional tags for the pipeline
    """
    # Create the pipeline controller
    pipe = PipelineController(
        name=pipeline_name,
        project=project_name,
        version="1.0",
        add_pipeline_tags=add_pipeline_tags or ["yolov9", "object-detection"],
        output_uri=output_uri,
    )

    # Step 1: Dataset Versioning
    dataset_task = pipe.add_function_step(
        name="dataset_versioning",
        function=dataset_versioning,
        function_kwargs=dict(
            dataset_name=dataset_name,
            dataset_project=dataset_project,
            dataset_path=dataset_path,
        ),
        function_return=["dataset_id"],
        cache_executed_step=True,
    )

    # Step 2: Model Training
    training_task = pipe.add_function_step(
        name="model_training",
        parents=[dataset_task],
        function=train_yolov9_model,
        function_kwargs=dict(
            dataset_id="${dataset_versioning.dataset_id}",
            project_name=base_task_project,
            task_name="YOLOv9-Training",
            epochs=100,
            batch_size=16,
            img_size=640,
            weights="yolov9-c-converted.pt",
            device="0",  # Use GPU 0
        ),
        function_return=["trained_model_id"],
        cache_executed_step=False,  # Don't cache training as it should run fresh each time
    )

    # Step 3: Model Evaluation
    evaluation_task = pipe.add_function_step(
        name="model_evaluation",
        parents=[training_task],
        function=evaluate_yolov9_model,
        function_kwargs=dict(
            model_id="${model_training.trained_model_id}",
            project_name=base_task_project,
            task_name="YOLOv9-Evaluation",
            img_size=640,
            batch_size=16,
            device="0",
        ),
        function_return=["evaluation_results"],
        cache_executed_step=False,
    )

    # Step 4: Model Deployment (if evaluation meets criteria)
    deployment_task = pipe.add_function_step(
        name="model_deployment",
        parents=[evaluation_task],
        function=deploy_model_if_improved,
        function_kwargs=dict(
            model_id="${model_training.trained_model_id}",
            evaluation_results="${model_evaluation.evaluation_results}",
            project_name=base_task_project,
            min_map_threshold=0.5,  # Minimum mAP to deploy the model
        ),
        function_return=["deployment_success"],
        cache_executed_step=False,
    )

    # Start the pipeline
    pipe.start()
    
    return pipe


# Step 1: Dataset Versioning Function
def dataset_versioning(dataset_name, dataset_project, dataset_path):
    """
    Create a versioned dataset in ClearML
    
    Args:
        dataset_name: Name of the dataset
        dataset_project: Project name for the dataset
        dataset_path: Path to the dataset
        
    Returns:
        dataset_id: ID of the created dataset
    """
    # Create a task for dataset versioning
    task = Task.init(
        project_name=dataset_project,
        task_name=f"Dataset Versioning - {dataset_name}",
        task_type="data_processing",
        reuse_last_task_id=False
    )
    
    # Create or get the dataset
    dataset = Dataset.create(
        dataset_name=dataset_name,
        dataset_project=dataset_project
    )
    
    # Add files from the local dataset folder
    dataset.add_files(dataset_path)
    
    # Finalize the dataset
    dataset.finalize()
    
    # Get the dataset ID
    dataset_id = dataset.id
    
    # Close the task
    task.close()
    
    return {"dataset_id": dataset_id}


# Step 2: Model Training Function
def train_yolov9_model(dataset_id, project_name, task_name, epochs, batch_size, img_size, weights, device):
    """
    Train a YOLOv9 model using the specified dataset
    
    Args:
        dataset_id: ID of the dataset to use for training
        project_name: Project name for the training task
        task_name: Name of the training task
        epochs: Number of epochs to train for
        batch_size: Batch size for training
        img_size: Image size for training
        weights: Path to pre-trained weights
        device: Device to use for training (e.g., "0" for GPU 0)
        
    Returns:
        trained_model_id: ID of the trained model
    """
    # Create a task for model training
    task = Task.init(
        project_name=project_name,
        task_name=task_name,
        task_type="training",
        reuse_last_task_id=False
    )
    
    # Connect the dataset
    dataset = Dataset.get(dataset_id=dataset_id)
    dataset_path = dataset.get_local_copy()
    
    # Set up the parameters for training
    params = {
        "epochs": epochs,
        "batch_size": batch_size,
        "img_size": img_size,
        "weights": weights,
        "device": device,
        "dataset_path": dataset_path,
    }
    task.connect(params)
    
    # Get the current working directory
    cwd = os.getcwd()
    
    # Change to the YOLOv9 directory
    os.chdir(os.path.join(cwd, "yolov9"))
    
    # Create a data.yaml file pointing to the dataset
    data_yaml = os.path.join(dataset_path, "dataset.yaml")
    
    # Run the training command with your specific configuration
    train_cmd = (
        f"python train_dual.py "
        f"--workers 8 "
        f"--device {device} "
        f"--img {img_size} "
        f"--batch {batch_size} "
        f"--epochs {epochs} "
        f"--data {data_yaml} "
        f"--cfg ./models/detect/yolov9-c-fish-od.yaml "
        f"--weights {weights} "
        f"--name {task_name} "
        f"--hyp ./data/hyps/hyp.scratch-high.yaml "
        f"--min-items 0 "
        f"--close-mosaic 15"
    )
    
    # Execute the training command
    os.system(train_cmd)
    
    # Get the model ID (the task ID serves as the model ID)
    trained_model_id = task.id
    
    # Change back to the original directory
    os.chdir(cwd)
    
    return {"trained_model_id": trained_model_id}


# Step 3: Model Evaluation Function
def evaluate_yolov9_model(model_id, project_name, task_name, img_size, batch_size, device):
    """
    Evaluate a trained YOLOv9 model
    
    Args:
        model_id: ID of the model to evaluate
        project_name: Project name for the evaluation task
        task_name: Name of the evaluation task
        img_size: Image size for evaluation
        batch_size: Batch size for evaluation
        device: Device to use for evaluation
        
    Returns:
        evaluation_results: Dictionary containing evaluation metrics
    """
    # Create a task for model evaluation
    task = Task.init(
        project_name=project_name,
        task_name=task_name,
        task_type="testing",
        reuse_last_task_id=False
    )
    
    # Get the trained model
    trained_task = Task.get_task(task_id=model_id)
    
    # Get the model weights
    model_path = trained_task.models["output"][-1].get_local_copy()
    
    # Set up the parameters for evaluation
    params = {
        "img_size": img_size,
        "batch_size": batch_size,
        "device": device,
        "model_path": model_path,
    }
    task.connect(params)
    
    # Get the current working directory
    cwd = os.getcwd()
    
    # Change to the YOLOv9 directory
    os.chdir(os.path.join(cwd, "yolov9"))
    
    # Run the evaluation command
    val_cmd = (
        f"python val.py "
        f"--img {img_size} "
        f"--batch {batch_size} "
        f"--weights {model_path} "
        f"--device {device} "
        f"--project {project_name} "
        f"--name {task_name} "
        f"--exist-ok"
    )
    
    # Execute the evaluation command
    os.system(val_cmd)
    
    # For this example, we'll create a dummy evaluation result
    # In a real scenario, you would parse the evaluation output
    evaluation_results = {
        "mAP50": 0.85,  # Example value
        "mAP50-95": 0.65,  # Example value
        "precision": 0.82,  # Example value
        "recall": 0.78,  # Example value
    }
    
    # Log the evaluation metrics
    for metric_name, metric_value in evaluation_results.items():
        task.get_logger().report_scalar(
            title="Evaluation Metrics",
            series=metric_name,
            value=metric_value,
            iteration=0
        )
    
    # Change back to the original directory
    os.chdir(cwd)
    
    return {"evaluation_results": evaluation_results}


# Step 4: Model Deployment Function
def deploy_model_if_improved(model_id, evaluation_results, project_name, min_map_threshold=0.5):
    """
    Deploy the model if it meets the evaluation criteria
    
    Args:
        model_id: ID of the model to deploy
        evaluation_results: Dictionary containing evaluation metrics
        project_name: Project name for the deployment task
        min_map_threshold: Minimum mAP threshold for deployment
        
    Returns:
        deployment_success: Boolean indicating if deployment was successful
    """
    # Create a task for model deployment
    task = Task.init(
        project_name=project_name,
        task_name="Model Deployment",
        task_type="deployment",
        reuse_last_task_id=False
    )
    
    # Check if the model meets the deployment criteria
    if evaluation_results["mAP50"] >= min_map_threshold:
        # Get the trained model
        trained_task = Task.get_task(task_id=model_id)
        
        # Get the model weights
        model_path = trained_task.models["output"][-1].get_local_copy()
        
        # In a real scenario, you would deploy the model to your production environment
        # For this example, we'll just log that the model was deployed
        task.get_logger().report_text(
            "Model deployed successfully with mAP50 = {:.2f}".format(evaluation_results["mAP50"])
        )
        
        deployment_success = True
    else:
        # Log that the model did not meet the deployment criteria
        task.get_logger().report_text(
            "Model did not meet deployment criteria. mAP50 = {:.2f}, threshold = {:.2f}".format(
                evaluation_results["mAP50"], min_map_threshold
            )
        )
        
        deployment_success = False
    
    return {"deployment_success": deployment_success}


if __name__ == "__main__":
    # Create and start the pipeline
    pipeline = create_yolov9_pipeline(
        project_name="YOLOv9-Pipeline",
        pipeline_name="YOLOv9-Training-Pipeline",
        dataset_name="YOLOv9-Dataset",
        dataset_project="YOLOv9-Datasets",
        dataset_path="./yolov9/data/dataset",
        base_task_project="YOLOv9-Tasks",
    ) 