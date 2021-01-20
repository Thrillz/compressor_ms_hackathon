from azureml.core import Workspace

# set up workspace
ws = Workspace.from_config()

# set up datastores
dstore = ws.datastores['compressor_datastore']

print('Workspace Name: ' + ws.name, 
      'Azure Region: ' + ws.location, 
      'Subscription Id: ' + ws.subscription_id, 
      'Resource Group: ' + ws.resource_group, 
      sep = '\n')

from azureml.core import Experiment

experiment = Experiment(ws, 'Compressor-Insight')

print('Experiment name: ' + experiment.name)

dataset_name = 'compressor_dataset'

from azureml.core.dataset import Dataset

dataset = Dataset.get_by_name(ws, name=dataset_name)
dataset_input = dataset.as_named_input(dataset_name)

from azureml.core import Environment
from azureml.core.conda_dependencies import CondaDependencies

train_env = Environment(name="many_models_environment")
train_conda_deps = CondaDependencies.create(pip_packages=['sklearn', 'pandas', 'joblib', 'matplotlib', 'azureml-core', 'azureml-dataprep[fuse]', 'numpy', 'scipy', 'azureml-defaults'])
train_env.python.conda_dependencies = train_conda_deps

cpu_cluster_name = "compressor-cpu"

from azureml.core.compute import AmlCompute

compute = AmlCompute(ws, cpu_cluster_name)

from azureml.pipeline.steps import ParallelRunConfig

processes_per_node = 3
node_count = 2
timeout = 1800

parallel_run_config = ParallelRunConfig(
    source_directory='./',
    entry_script='main_test.py',
    mini_batch_size="1",
    run_invocation_timeout=timeout,
    error_threshold=10,
    output_action="append_row",
    environment=train_env,
    process_count_per_node=processes_per_node,
    compute_target=compute,
    node_count=node_count)

from azureml.data import OutputFileDatasetConfig

output_dataset = OutputFileDatasetConfig(name='batch_results', destination=( dstore, 'train-results/{​​run-id}​​'))
from azureml.pipeline.steps import ParallelRunStep


parallel_run_step = ParallelRunStep(
    name="many-models-training",
    parallel_run_config=parallel_run_config,
    inputs=[dataset_input],
    output=output_dir,
    allow_reuse=False,
    arguments=['--output_path', output_dataset]
)

from azureml.pipeline.core import Pipeline

pipeline = Pipeline(workspace=ws, steps=[parallel_run_step])
run = experiment.submit(pipeline)

#Wait for the run to complete
run.wait_for_completion(show_output=False, raise_on_error=True)



