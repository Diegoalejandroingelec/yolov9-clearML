"""
Hyperparameter Optimization for YOLOv9 using ClearML
This script creates a hyperparameter optimization task for YOLOv9 training.
"""

import os
from clearml import Task
from clearml.automation import HyperParameterOptimizer
from clearml.automation.optuna import OptimizerOptuna
from clearml.automation.parameters import UniformParameterRange, DiscreteParameterRange

def create_yolov9_hpo_task(
    project_name="YOLOv9-HPO",
    task_name="YOLOv9-Hyperparameter-Optimization",
    dataset_id=None,
    base_task_id=None,
    total_max_jobs=10,
    max_concurrent_jobs=2,
    execution_queue="default",
    optimization_time_limit=None,
):
    """
    Create a hyperparameter optimization task for YOLOv9
    
    Args:
        project_name: Name of the project
        task_name: Name of the task
        dataset_id: ID of the dataset to use
        base_task_id: ID of the base task to clone for optimization
        total_max_jobs: Maximum number of jobs to run
        max_concurrent_jobs: Maximum number of concurrent jobs
        execution_queue: Queue to use for execution
        optimization_time_limit: Time limit for optimization in minutes
        
    Returns:
        optimizer: The hyperparameter optimizer
    """
    # Create the base task for hyperparameter optimization
    task = Task.init(
        project_name=project_name,
        task_name=task_name,
        task_type="optimizer",
        reuse_last_task_id=False
    )
    
    # Get the base task to optimize
    if not base_task_id:
        raise ValueError("base_task_id must be provided")
    
    # Configure the hyperparameter optimization
    optimizer = HyperParameterOptimizer(
        base_task_id=base_task_id,
        hyper_parameters=[
            # Learning rate
            UniformParameterRange(
                "Args/lr0", min_value=1e-5, max_value=1e-2, step_size=1e-5
            ),
            # Batch size
            DiscreteParameterRange(
                "Args/batch_size", values=[8, 16, 32, 64]
            ),
            # Image size
            DiscreteParameterRange(
                "Args/img_size", values=[416, 512, 640, 768]
            ),
            # Weight decay
            UniformParameterRange(
                "Args/weight_decay", min_value=1e-5, max_value=1e-2, step_size=1e-5
            ),
            # Momentum
            UniformParameterRange(
                "Args/momentum", min_value=0.8, max_value=0.99, step_size=0.01
            ),
        ],
        objective_metric_title="metrics",
        objective_metric_series="mAP_0.5",
        objective_metric_sign="max",  # We want to maximize mAP
        max_number_of_concurrent_tasks=max_concurrent_jobs,
        optimizer_class=OptimizerOptuna,
        execution_queue=execution_queue,
        total_max_jobs=total_max_jobs,
        optimization_time_limit=optimization_time_limit,
        save_top_k_tasks_only=5,  # Save only the top 5 tasks
    )
    
    # If we have a dataset ID, we need to connect it to all the tasks
    if dataset_id:
        optimizer.set_task_parameters(
            name="dataset_id",
            value=dataset_id,
        )
    
    # Start the optimization
    optimizer.start()
    
    # Return the optimizer for reference
    return optimizer


def clone_task_for_hpo(
    base_task_id,
    project_name="YOLOv9-HPO",
    task_name="YOLOv9-Base-Task-for-HPO",
):
    """
    Clone a task for hyperparameter optimization
    
    Args:
        base_task_id: ID of the base task to clone
        project_name: Name of the project
        task_name: Name of the task
        
    Returns:
        cloned_task_id: ID of the cloned task
    """
    # Get the base task
    base_task = Task.get_task(task_id=base_task_id)
    
    # Clone the task
    cloned_task = Task.clone(
        source_task=base_task,
        name=task_name,
        project=project_name,
    )
    
    # Modify the cloned task to make it suitable for HPO
    # For example, reduce the number of epochs for faster iterations
    cloned_task.set_parameter("Args/epochs", 10)
    
    # Close the task to save the changes
    cloned_task.close()
    
    return cloned_task.id


if __name__ == "__main__":
    # Example usage:
    # 1. First, you need to have a base task that you want to optimize
    # This could be a training task from your pipeline
    
    # 2. Clone the task for HPO
    # base_task_id = "your_training_task_id"
    # cloned_task_id = clone_task_for_hpo(base_task_id)
    
    # 3. Create and start the HPO
    # dataset_id = "your_dataset_id"
    # optimizer = create_yolov9_hpo_task(
    #     base_task_id=cloned_task_id,
    #     dataset_id=dataset_id,
    #     total_max_jobs=20,
    #     max_concurrent_jobs=4,
    # )
    
    print("To use this script, uncomment the example code and replace the placeholder IDs with your actual task and dataset IDs.")
    print("You can get these IDs from the ClearML web UI or from your pipeline execution.") 