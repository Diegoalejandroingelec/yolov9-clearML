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
            project_name=base_task_project,
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
            project_name=base_task_project,
            task_name="YOLOv9-Get-Best-Hyperparameters",
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
        execution_queue=worker_queue,
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
    return dataset_id


# Step 2: Create Base Training Task
def create_base_training_task(dataset_id,
                              project_name,
                              task_name,
                              epochs,
                              batch_size,
                              img_size,
                              weights,
                              device):
    """
    Create a base training task for hyperparameter optimization.
    """
    print("Creating base training task...", flush=True)
    
    print(f"Initializing ClearML training task for project: {project_name}, task: {task_name}", flush=True)
    task = Task.init(
        project_name=project_name,
        task_name=task_name,
        task_type="training",
        reuse_last_task_id=False
    )
    
    print(f"Received Dataset ID: {dataset_id}", flush=True)
    
    print("Fetching dataset from ClearML...", flush=True)
    dataset = Dataset.get(dataset_id=dataset_id)
    
    print("Retrieving local copy of dataset...", flush=True)
    dataset_path = dataset.get_local_copy()
    print(f"Local dataset path: {dataset_path}", flush=True)
    
    params = {
        "Args/epochs": epochs,
        "Args/batch_size": batch_size,
        "Args/img_size": img_size,
        "Args/weights": weights,
        "Args/device": device,
        "Args/dataset_path": dataset_path,
    }
    print(f"Connecting parameters to the task: {params}", flush=True)
    task.connect(params)
    
    print("Closing task...", flush=True)
    task.close()
    
    base_task_id = task.id
    print(f"Base training task created successfully with Task ID: {base_task_id}", flush=True)
    return base_task_id


# Step 5: Get Best Hyperparameters
def get_best_hyperparameters(optimizer,
                             project_name,
                             task_name):
    """
    Get the best hyperparameters from the optimizer.
    """
    print("Initializing task for getting best hyperparameters...", flush=True)
    task = Task.init(
        project_name=project_name,
        task_name=task_name,
        task_type="data_processing",
        reuse_last_task_id=False
    )
    
    print("Waiting for optimizer to finish...", flush=True)
    optimizer.wait()
    
    print("Retrieving top experiment from optimizer...", flush=True)
    top_exp = optimizer.get_top_experiments(top_k=1)[0]
    print("Top experiment retrieved.", flush=True)
    
    best_hyperparameters = {
        "lr0": top_exp.get_parameter_value("Args/lr0"),
        "batch_size": top_exp.get_parameter_value("Args/batch_size"),
        "img_size": top_exp.get_parameter_value("Args/img_size"),
        "weight_decay": top_exp.get_parameter_value("Args/weight_decay"),
        "momentum": top_exp.get_parameter_value("Args/momentum"),
    }
    print(f"Best hyperparameters: {best_hyperparameters}", flush=True)
    
    report_text = "Best hyperparameters found:\n" + "\n".join([f"{k}: {v}" for k, v in best_hyperparameters.items()])
    print(report_text, flush=True)
    task.get_logger().report_text(report_text)
    
    print("Retrieving performance metrics for the best configuration...", flush=True)
    metrics = top_exp.get_last_metrics()
    metrics_report = (
        "Performance metrics of best configuration:\n" +
        f"mAP50: {metrics['metrics/mAP_0.5']}\n" +
        f"mAP50-95: {metrics['metrics/mAP_0.5_0.95']}"
    )
    print(metrics_report, flush=True)
    task.get_logger().report_text(metrics_report)
    
    print("Closing the best hyperparameters task...", flush=True)
    task.close()
    
    return best_hyperparameters


# Step 6: Model Training with Best Hyperparameters
def train_yolov9_model_with_best_params(dataset_id, project_name, task_name, best_hyperparameters, epochs, weights, device):
    """
    Train a YOLOv9 model using the best hyperparameters.
    """
    print("Initializing training task...", flush=True)
    task = Task.init(
        project_name=project_name,
        task_name=task_name,
        task_type="training",
        reuse_last_task_id=False
    )
    
    print(f"Fetching dataset with ID: {dataset_id}", flush=True)
    dataset = Dataset.get(dataset_id=dataset_id)
    print("Retrieving local copy of the dataset...", flush=True)
    dataset_path = dataset.get_local_copy()
    print(f"Dataset local path: {dataset_path}", flush=True)
    
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
    print("Connecting parameters to the task:", params, flush=True)
    task.connect(params)
    
    cwd = os.getcwd()
    print(f"Current working directory: {cwd}", flush=True)
    yolov9_dir = os.path.join(cwd, "yolov9")
    print(f"Changing directory to: {yolov9_dir}", flush=True)
    os.chdir(yolov9_dir)
    
    data_yaml = os.path.join(dataset_path, "dataset.yaml")
    custom_hyp_file = os.path.join(cwd, "yolov9", "data", "hyps", f"hyp.custom.{task_name}.yaml")
    print(f"Data yaml path: {data_yaml}", flush=True)
    print(f"Custom hyperparameter file will be written to: {custom_hyp_file}", flush=True)
    
    print("Loading base hyperparameter configuration from './data/hyps/hyp.scratch-high.yaml'...", flush=True)
    with open("./data/hyps/hyp.scratch-high.yaml", "r") as f:
        hyp_config = yaml.safe_load(f)
    
    print("Updating hyperparameter configuration with best hyperparameters...", flush=True)
    hyp_config.update({
        "lr0": lr0,
        "weight_decay": weight_decay,
        "momentum": momentum,
    })
    
    print("Writing updated hyperparameter configuration to custom hyp file...", flush=True)
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
    print(f"Executing training command:\n{train_cmd}", flush=True)
    os.system(train_cmd)
    
    trained_model_id = task.id
    print(f"Training task completed. Task ID (trained model id): {trained_model_id}", flush=True)
    
    print(f"Changing back to original directory: {cwd}", flush=True)
    os.chdir(cwd)
    
    return trained_model_id


# Step 7: Model Evaluation Function
def evaluate_yolov9_model(model_id, project_name, task_name, img_size, batch_size, device):
    """
    Evaluate a trained YOLOv9 model.
    """
    print("Initializing evaluation task...", flush=True)
    task = Task.init(
        project_name=project_name,
        task_name=task_name,
        task_type="testing",
        reuse_last_task_id=False
    )
    
    print(f"Fetching trained task with model ID: {model_id}", flush=True)
    trained_task = Task.get_task(task_id=model_id)
    
    print("Retrieving model from trained task's output artifacts...", flush=True)
    model_output_list = trained_task.models["output"]
    print(f"Number of model outputs found: {len(model_output_list)}", flush=True)
    
    model_path = model_output_list[-1].get_local_copy()
    print(f"Model local path: {model_path}", flush=True)
    
    params = {
        "img_size": img_size,
        "batch_size": batch_size,
        "device": device,
        "model_path": model_path,
    }
    print("Connecting parameters to evaluation task:", params, flush=True)
    task.connect(params)
    
    cwd = os.getcwd()
    print(f"Current working directory: {cwd}", flush=True)
    yolov9_dir = os.path.join(cwd, "yolov9")
    print(f"Changing directory to: {yolov9_dir}", flush=True)
    os.chdir(yolov9_dir)
    
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
    print(f"Executing validation command:\n{val_cmd}", flush=True)
    
    import subprocess
    process = subprocess.Popen(
        val_cmd.split(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True
    )
    stdout, stderr = process.communicate()
    
    print("Validation command completed.", flush=True)
    print("Parsing evaluation results...", flush=True)
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
    
    print("Evaluation results parsed:", evaluation_results, flush=True)
    
    for metric_name, metric_value in evaluation_results.items():
        print(f"Reporting metric: {metric_name} with value: {metric_value}", flush=True)
        task.get_logger().report_scalar(
            title="Evaluation Metrics",
            series=metric_name,
            value=metric_value,
            iteration=0
        )
    
    print("Reporting validation output and errors (if any)...", flush=True)
    task.get_logger().report_text("Validation Output:\n" + stdout)
    if stderr:
        task.get_logger().report_text("Validation Errors:\n" + stderr)
    
    print("Changing back to original directory...", flush=True)
    os.chdir(cwd)
    print("Evaluation task completed.", flush=True)
    
    return evaluation_results


# Step 8: Model Deployment Function
def deploy_model_if_improved(model_id, evaluation_results, project_name, min_map_threshold=0.5):
    """
    Deploy the model if it meets the evaluation criteria.
    """
    print("Initializing deployment task...", flush=True)
    task = Task.init(
        project_name=project_name,
        task_name="Model Deployment",
        task_type="deployment",
        reuse_last_task_id=False
    )
    
    mAP50 = evaluation_results.get("mAP50", 0)
    print(f"Evaluation results received: mAP50 = {mAP50}, threshold = {min_map_threshold}", flush=True)
    
    if mAP50 >= min_map_threshold:
        print("Deployment criteria met. Proceeding to deploy the model...", flush=True)
        trained_task = Task.get_task(task_id=model_id)
        print("Fetching model output from the trained task...", flush=True)
        model_output_list = trained_task.models["output"]
        print(f"Number of model outputs: {len(model_output_list)}", flush=True)
        model_path = model_output_list[-1].get_local_copy()
        print(f"Retrieved model local path: {model_path}", flush=True)
        
        deploy_message = "Model deployed successfully with mAP50 = {:.2f}".format(mAP50)
        print(deploy_message, flush=True)
        task.get_logger().report_text(deploy_message)
        deployment_success = True
    else:
        not_deployed_message = "Model did not meet deployment criteria. mAP50 = {:.2f}, threshold = {:.2f}".format(
            mAP50, min_map_threshold
        )
        print(not_deployed_message, flush=True)
        task.get_logger().report_text(not_deployed_message)
        deployment_success = False
    
    print("Deployment task completed.", flush=True)
    return deployment_success


def create_yolov9_hpo_task(
    project_name="YOLOv9-HPO",
    task_name="YOLOv9-Hyperparameter-Optimization",
    dataset_id=None,
    base_task_id=None,
    total_max_jobs=10,
    max_concurrent_jobs=2,
    execution_queue="default",
    optimization_time_limit=None,
    max_iteration_per_job=10,  # New parameter added with a default value
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
        max_iteration_per_job: Maximum number of iterations each job will run
        
    Returns:
        optimizer: The hyperparameter optimizer
    """
    print("Initializing hyperparameter optimization task...", flush=True)
    task = Task.init(
        project_name=project_name,
        task_name=task_name,
        task_type="optimizer",
        reuse_last_task_id=False
    )
    print(f"Optimizer task initialized with ID: {task.id}", flush=True)
    
    if not base_task_id:
        raise ValueError("base_task_id must be provided")
    print(f"Base task ID for optimization: {base_task_id}", flush=True)
    
    print("Configuring hyperparameter optimization with the following parameters:", flush=True)
    print(f"Total max jobs: {total_max_jobs}", flush=True)
    print(f"Max concurrent jobs: {max_concurrent_jobs}", flush=True)
    print(f"Execution queue: {execution_queue}", flush=True)
    print(f"Optimization time limit: {optimization_time_limit}", flush=True)
    print(f"Max iterations per job: {max_iteration_per_job}", flush=True)
    
    optimizer = HyperParameterOptimizer(
        base_task_id=base_task_id,
        hyper_parameters=[
            UniformParameterRange("Args/lr0", min_value=1e-5, max_value=1e-2, step_size=1e-5),
            DiscreteParameterRange("Args/batch_size", values=[8, 16, 32, 64]),
            DiscreteParameterRange("Args/img_size", values=[416, 512, 640, 768]),
            UniformParameterRange("Args/weight_decay", min_value=1e-5, max_value=1e-2, step_size=1e-5),
            UniformParameterRange("Args/momentum", min_value=0.8, max_value=0.99, step_size=0.01),
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
        max_iteration_per_job=max_iteration_per_job,  # Provide the missing parameter
    )
    print("Hyperparameter optimizer configured.", flush=True)
    
    # if dataset_id:
    #     print(f"Setting dataset_id parameter: {dataset_id}", flush=True)
    #     optimizer.set_task_parameters(
    #         name="dataset_id",
    #         value=dataset_id,
    #     )
    
    print("Starting hyperparameter optimization...", flush=True)
    optimizer.start()
    print("Hyperparameter optimization started successfully.", flush=True)
    
    return optimizer



def clone_task_for_hpo(
    base_task_id,
    project_name,
    task_name
):
    """
    Clone a task for hyperparameter optimization.

    Args:
        base_task_id: ID of the base task to clone.
        project_name: project name where the cloned task should be created.
        task_name: Name of the cloned task.
        
    Returns:
        cloned_task_id: ID of the cloned task.
    """

    def get_or_create_project_id(project_name):
        """
        Create a temporary task with the given project name to get its project id.
        This will auto-create the project if it doesn't exist.
        """
        # Create a temporary task. This will cause ClearML to create the project if needed.
        temp_task = Task.init(
            project_name=project_name,
            task_name="Temporary Project Lookup",
            task_type="data_processing",
            reuse_last_task_id=False
        )
        project_id = temp_task.get_project_id(project_name)
        print(f"Using project '{project_name}' with id: {project_id}", flush=True)
        temp_task.close()
        return project_id

    project_id = get_or_create_project_id(project_name)
    
    print(f"Base task ID: {base_task_id}", flush=True)
    print(f"Project ID: {project_id}", flush=True)
    print(f"Task name: {task_name}", flush=True)
    
    base_task = Task.get_task(task_id=base_task_id)
    cloned_task = Task.clone(
        source_task=base_task,
        name=task_name,
        project=project_id,
    )
    # For faster iterations during HPO, we reduce the epochs.
    cloned_task.set_parameter("Args/epochs", 10)
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
