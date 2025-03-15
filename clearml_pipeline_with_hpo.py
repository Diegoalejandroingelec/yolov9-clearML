"""
Enhanced ClearML Pipeline for YOLOv9 Training with Hyperparameter Optimization
This script creates a complete pipeline that includes:
1. Dataset versioning and management
2. Hyperparameter optimization
3. Model training with optimal hyperparameters
4. Model evaluation
5. Model deployment
"""

import os
import sys
from pathlib import Path
from clearml import Task, Dataset, PipelineController
from clearml.automation import HyperParameterOptimizer
from clearml.automation.optuna import OptimizerOptuna
from clearml.automation.parameters import UniformParameterRange, DiscreteParameterRange
import yaml


# Define the enhanced pipeline
def create_yolov9_pipeline_with_hpo(
    project_name="YOLOv9-Pipeline-HPO",
    pipeline_name="YOLOv9-Training-Pipeline-with-HPO",
    dataset_name="YOLOv9-Dataset",
    dataset_project="YOLOv9-Datasets",
    dataset_path="/home/diego/Documents/master/S4/AI_studio/yolov9-clearML/yolov9/data/dataset",
    base_task_project="YOLOv9-Tasks",
    hpo_project="YOLOv9-HPO",
    output_uri=None,
    add_pipeline_tags=None,
    hpo_max_jobs=10,
    hpo_concurrent_jobs=2,
    pipeline_queue="pipeline-controller-queue",
    worker_queue="worker-tasks-queue",
):
    """
    Create a ClearML pipeline for YOLOv9 training with hyperparameter optimization
    
    Args:
        pipeline_queue: The queue for the main pipeline controller
        worker_queue: The queue where child tasks will run
    """
    # Create the pipeline controller
    pipe = PipelineController(
        name=pipeline_name,
        project=project_name,
        version="1.0",
        add_pipeline_tags=add_pipeline_tags or ["yolov9", "object-detection", "hpo"],
        output_uri=output_uri
    )
    
    # Set the execution queue for the pipeline itself
    pipe.set_default_execution_queue(pipeline_queue)

    # Step 1: Dataset Versioning (runs on worker queue)
    pipe.add_function_step(
        name="dataset_versioning",
        function=dataset_versioning,
        function_kwargs=dict(
            dataset_name=dataset_name,
            dataset_project=dataset_project,
            dataset_path=dataset_path,
        ),
        function_return=["dataset_id"],
        cache_executed_step=False,
        execution_queue=worker_queue,  # Run on worker queue
    )

    # Step 2: Create Base Training Task for HPO (worker queue)
    pipe.add_function_step(
        name="create_base_task",
        parents=["dataset_versioning"],
        function=create_base_training_task,
        function_kwargs=dict(
            dataset_id="${dataset_versioning.dataset_id}",
            project_name=base_task_project,
            task_name="YOLOv9-Base-Training",
            epochs=5,
            batch_size=16,
            img_size=640,
            weights="yolov9-c-converted.pt",
            device="0",
        ),
        function_return=["base_task_id"],
        cache_executed_step=False,
        execution_queue=worker_queue,
    )

    # Step 3: Clone Base Task for HPO (worker queue)
    pipe.add_function_step(
        name="clone_task_for_hpo",
        parents=["create_base_task"],
        function=clone_task_for_hpo,
        function_kwargs=dict(
            base_task_id="${create_base_task.base_task_id}",
            project_name=hpo_project,
            task_name="YOLOv9-Base-Task-for-HPO",
        ),
        function_return=["cloned_task_id"],
        cache_executed_step=False,
        execution_queue=worker_queue,
    )

    # Step 4: Hyperparameter Optimization (worker queue)
    pipe.add_function_step(
        name="hyperparameter_optimization",
        parents=["clone_task_for_hpo", "dataset_versioning"],
        function=create_yolov9_hpo_task,
        function_kwargs=dict(
            project_name=hpo_project,
            task_name="YOLOv9-Hyperparameter-Optimization",
            dataset_id="${dataset_versioning.dataset_id}",
            base_task_id="${clone_task_for_hpo.cloned_task_id}",
            total_max_jobs=hpo_max_jobs,
            max_concurrent_jobs=hpo_concurrent_jobs,
        ),
        function_return=["optimizer"],
        cache_executed_step=False,
        execution_queue=worker_queue,
    )

    # Step 5: Get Best Hyperparameters (worker queue)
    pipe.add_function_step(
        name="get_best_hyperparameters",
        parents=["hyperparameter_optimization"],
        function=get_best_hyperparameters,
        function_kwargs=dict(
            optimizer="${hyperparameter_optimization.optimizer}",
        ),
        function_return=["best_hyperparameters"],
        cache_executed_step=False,
        execution_queue=worker_queue,
    )

    # Step 6: Model Training with Best Hyperparameters (worker queue)
    pipe.add_function_step(
        name="model_training",
        parents=["get_best_hyperparameters", "dataset_versioning"],
        function=train_yolov9_model_with_best_params,
        function_kwargs=dict(
            dataset_id="${dataset_versioning.dataset_id}",
            project_name=base_task_project,
            task_name="YOLOv9-Training-Best-Params",
            best_hyperparameters="${get_best_hyperparameters.best_hyperparameters}",
            epochs=100,
            weights="yolov9-c-converted.pt",
            device="0",
        ),
        function_return=["trained_model_id"],
        cache_executed_step=False,
        execution_queue=worker_queue,
    )

    # Step 7: Model Evaluation (worker queue)
    pipe.add_function_step(
        name="model_evaluation",
        parents=["model_training"],
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
        execution_queue=worker_queue,
    )

    # Step 8: Model Deployment (pipeline queue, since it involves decision-making)
    pipe.add_function_step(
        name="model_deployment",
        parents=["model_evaluation"],
        function=deploy_model_if_improved,
        function_kwargs=dict(
            model_id="${model_training.trained_model_id}",
            evaluation_results="${model_evaluation.evaluation_results}",
            project_name=base_task_project,
            min_map_threshold=0.5,
        ),
        function_return=["deployment_success"],
        cache_executed_step=False,
        execution_queue=worker_queue,  # Deployment runs in the pipeline queue
    )

    # Start the pipeline
    pipe.start(queue=pipeline_queue)

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
    task = Task.init(
        project_name=dataset_project,
        task_name=f"Dataset Versioning - {dataset_name}",
        task_type="data_processing",
        reuse_last_task_id=False
    )
    
    dataset = Dataset.create(
        dataset_name=dataset_name,
        dataset_project=dataset_project
    )
    
    print("Adding dataset files...")
    dataset.add_files(dataset_path)

    print("Uploading dataset files to ClearML storage...")
    dataset.upload()  # Ensure files are uploaded before finalizing

    print("Finalizing dataset version...")
    dataset.finalize()
    
    dataset_id = dataset.id

    # Store dataset_id as an artifact so it can be used later
    task.upload_artifact(name="dataset_id", artifact_object=dataset_id)

    task.close()
    
    print(f"Dataset versioning completed. Dataset ID: {dataset_id}")
    return {"dataset_id": dataset_id}


# Step 2: Create Base Training Task
def create_base_training_task(dataset_id, project_name, task_name, epochs, batch_size, img_size, weights, device):
    """
    Create a base training task for hyperparameter optimization.
    """
    task = Task.init(
        project_name=project_name,
        task_name=task_name,
        task_type="training",
        reuse_last_task_id=False
    )
    
    # Directly use the dataset_id passed in from the pipeline
    dataset = Dataset.get(dataset_id=dataset_id)
    dataset_path = dataset.get_local_copy()
    
    params = {
        "Args/epochs": epochs,
        "Args/batch_size": batch_size,
        "Args/img_size": img_size,
        "Args/weights": weights,
        "Args/device": device,
        "Args/dataset_path": dataset_path,
    }
    task.connect(params)
    task.close()
    
    return {"base_task_id": task.id}




# Step 5: Get Best Hyperparameters
def get_best_hyperparameters(optimizer):
    """
    Get the best hyperparameters from the optimizer.
    """
    task = Task.init(
        project_name="YOLOv9-HPO",
        task_name="Get-Best-Hyperparameters",
        task_type="data_processing",
        reuse_last_task_id=False
    )
    
    optimizer.wait()
    top_exp = optimizer.get_top_experiments(top_k=1)[0]
    
    best_hyperparameters = {
        "lr0": top_exp.get_parameter_value("Args/lr0"),
        "batch_size": top_exp.get_parameter_value("Args/batch_size"),
        "img_size": top_exp.get_parameter_value("Args/img_size"),
        "weight_decay": top_exp.get_parameter_value("Args/weight_decay"),
        "momentum": top_exp.get_parameter_value("Args/momentum"),
    }
    
    task.get_logger().report_text(
        "Best hyperparameters found:\n" +
        "\n".join([f"{k}: {v}" for k, v in best_hyperparameters.items()])
    )
    
    metrics = top_exp.get_last_metrics()
    task.get_logger().report_text(
        "Performance metrics of best configuration:\n" +
        f"mAP50: {metrics['metrics/mAP_0.5']}\n" +
        f"mAP50-95: {metrics['metrics/mAP_0.5_0.95']}"
    )
    
    task.close()
    return {"best_hyperparameters": best_hyperparameters}


# Step 6: Model Training with Best Hyperparameters
def train_yolov9_model_with_best_params(dataset_id, project_name, task_name, best_hyperparameters, epochs, weights, device):
    """
    Train a YOLOv9 model using the best hyperparameters.
    """
    task = Task.init(
        project_name=project_name,
        task_name=task_name,
        task_type="training",
        reuse_last_task_id=False
    )
    
    dataset = Dataset.get(dataset_id=dataset_id)
    dataset_path = dataset.get_local_copy()
    
    lr0 = best_hyperparameters.get("lr0", 0.01)
    batch_size = best_hyperparameters.get("batch_size", 16)
    img_size = best_hyperparameters.get("img_size", 640)
    weight_decay = best_hyperparameters.get("weight_decay", 0.0005)
    momentum = best_hyperparameters.get("momentum", 0.937)
    
    params = {
        "epochs": epochs,
        "batch_size": batch_size,
        "img_size": img_size,
        "weights": weights,
        "device": device,
        "dataset_path": dataset_path,
        "lr0": lr0,
        "weight_decay": weight_decay,
        "momentum": momentum,
    }
    task.connect(params)
    
    cwd = os.getcwd()
    os.chdir(os.path.join(cwd, "yolov9"))
    
    data_yaml = os.path.join(dataset_path, "dataset.yaml")
    custom_hyp_file = os.path.join(cwd, "yolov9", "data", "hyps", f"hyp.custom.{task_name}.yaml")
    with open("./data/hyps/hyp.scratch-high.yaml", "r") as f:
        hyp_config = yaml.safe_load(f)
    
    hyp_config.update({
        "lr0": lr0,
        "weight_decay": weight_decay,
        "momentum": momentum,
    })
    
    with open(custom_hyp_file, "w") as f:
        yaml.dump(hyp_config, f)
    
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
        f"--hyp {custom_hyp_file} "
        f"--min-items 0 "
        f"--close-mosaic 15"
    )
    
    os.system(train_cmd)
    trained_model_id = task.id
    os.chdir(cwd)
    
    return {"trained_model_id": trained_model_id}


# Step 7: Model Evaluation Function
def evaluate_yolov9_model(model_id, project_name, task_name, img_size, batch_size, device):
    """
    Evaluate a trained YOLOv9 model.
    """
    task = Task.init(
        project_name=project_name,
        task_name=task_name,
        task_type="testing",
        reuse_last_task_id=False
    )
    
    trained_task = Task.get_task(task_id=model_id)
    model_path = trained_task.models["output"][-1].get_local_copy()
    
    params = {
        "img_size": img_size,
        "batch_size": batch_size,
        "device": device,
        "model_path": model_path,
    }
    task.connect(params)
    
    cwd = os.getcwd()
    os.chdir(os.path.join(cwd, "yolov9"))
    
    val_cmd = (
        f"python val.py "
        f"--img {img_size} "
        f"--batch {batch_size} "
        f"--weights {model_path} "
        f"--device {device} "
        f"--task test "  
        f"--name {task_name} "
        f"--exist-ok"
    )
    
    import subprocess
    process = subprocess.Popen(
        val_cmd.split(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True
    )
    stdout, stderr = process.communicate()
    
    evaluation_results = {}
    for line in stdout.split('\n'):
        if 'all' in line and 'Average Precision' in line:
            parts = line.split()
            for i, part in enumerate(parts):
                if '@0.5' in part:
                    evaluation_results['mAP50'] = float(parts[i+1])
                elif '@0.5:0.95' in part:
                    evaluation_results['mAP50-95'] = float(parts[i+1])
        elif 'all' in line and 'Precision' in line:
            parts = line.split()
            evaluation_results['precision'] = float(parts[-1])
        elif 'all' in line and 'Recall' in line:
            parts = line.split()
            evaluation_results['recall'] = float(parts[-1])
    
    for metric_name, metric_value in evaluation_results.items():
        task.get_logger().report_scalar(
            title="Evaluation Metrics",
            series=metric_name,
            value=metric_value,
            iteration=0
        )
    
    task.get_logger().report_text("Validation Output:\n" + stdout)
    if stderr:
        task.get_logger().report_text("Validation Errors:\n" + stderr)
    
    os.chdir(cwd)
    return {"evaluation_results": evaluation_results}


# Step 8: Model Deployment Function
def deploy_model_if_improved(model_id, evaluation_results, project_name, min_map_threshold=0.5):
    """
    Deploy the model if it meets the evaluation criteria.
    """
    task = Task.init(
        project_name=project_name,
        task_name="Model Deployment",
        task_type="deployment",
        reuse_last_task_id=False
    )
    
    if evaluation_results["mAP50"] >= min_map_threshold:
        trained_task = Task.get_task(task_id=model_id)
        model_path = trained_task.models["output"][-1].get_local_copy()
        task.get_logger().report_text(
            "Model deployed successfully with mAP50 = {:.2f}".format(evaluation_results["mAP50"])
        )
        deployment_success = True
    else:
        task.get_logger().report_text(
            "Model did not meet deployment criteria. mAP50 = {:.2f}, threshold = {:.2f}".format(
                evaluation_results["mAP50"], min_map_threshold
            )
        )
        deployment_success = False
    
    return {"deployment_success": deployment_success}



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
    pipeline = create_yolov9_pipeline_with_hpo(
        project_name="YOLOv9-Pipeline-HPO",
        pipeline_name="YOLOv9-Training-Pipeline-with-HPO",
        dataset_name="YOLOv9-Dataset",
        dataset_project="YOLOv9-Datasets",
        dataset_path="/home/diego/Documents/master/S4/AI_studio/yolov9-clearML/yolov9/data/dataset",
        base_task_project="YOLOv9-Tasks",
        hpo_project="YOLOv9-HPO",
        hpo_max_jobs=10,
        hpo_concurrent_jobs=2,
        pipeline_queue="pipeline-controller-queue",
        worker_queue="worker-tasks-queue",
    )
