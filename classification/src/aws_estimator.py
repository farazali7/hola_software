import sagemaker
from sagemaker.pytorch import PyTorch

# Initializes SageMaker session which holds context data
sagemaker_session = sagemaker.Session()

# The bucket containing our input data
bucket = 's3://sagemaker-us-east-2-910185381360/iter1/data'

# The IAM Role which SageMaker will impersonate to run the estimator
# Remember you cannot use sagemaker.get_execution_role()
# if you're not in a SageMaker notebook, an EC2 or a Lambda
# (i.e. running from your local PC)

role = 'arn:aws:iam::910185381360:role/fali_trainer'

# Create a new PyTorch Estimator with params
estimator = PyTorch(
  # name of the runnable script containing __main__ function (entrypoint)
  entry_point='aws_train.py',
  # path of the folder containing training code. It could also contain a
  # requirements.txt file with all the dependencies that needs
  # to be installed before running
  source_dir='./src',
  role=role,
  framework_version='1.13.1',
  py_version="py39",
  instance_count=1,
  instance_type='ml.g4dn.xlarge',
  # these hyperparameters are passed to the main script as arguments and
  # can be overridden when fine tuning the algorithm
)

estimator.fit({'train': bucket})
