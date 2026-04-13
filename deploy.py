import os
import tarfile

import sagemaker
import torch
from dotenv import load_dotenv
from sagemaker.deserializers import JSONDeserializer
from sagemaker.pytorch import PyTorchModel
from sagemaker.serializers import JSONSerializer

load_dotenv()


print("Initial Setup...")
DEPLOY_DEVICE = "ml.m5.large"
TAR_NAME = "model.tar.gz"

ARN = os.getenv("IAM")
print(f"Deploying on {DEPLOY_DEVICE}")


print("Setting up the device...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"running on {device}")

# save model to a directory
print("Saving the model...")
LOCAL_MODEL_DIR: str = "model"
os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)  # prep the path for the model
model_path = os.path.join(LOCAL_MODEL_DIR, "drfrond.pth")

code_dir = os.path.join(LOCAL_MODEL_DIR, "code")  # prep the path for the package
os.makedirs(code_dir, exist_ok=True)

# already have inference.py in /model/code folder
# if os.path.exists('src/inference.py'):
#     shutil.copy('src/inference.py', os.path.join(code_dir, 'inference.py'))

with tarfile.open(TAR_NAME, "w:gz") as tar:  # create the archive
    tar.add(model_path, arcname="model.pth")
    tar.add(code_dir, arcname="code")

print(f"Saved model to {TAR_NAME}")


print("Uploading to S3...")
try:
    session = sagemaker.Session()
    try:
        role = sagemaker.get_execution_role()
    except (ValueError, RuntimeError):  # if not running on SageMaker
        role = ARN
    bucket = session.default_bucket()
    print(f"Bucket: {bucket}")
except Exception as e:
    print(e)
    exit(1)

s3_prefix = "DrFrond"
s3_model_path = session.upload_data(path=TAR_NAME, bucket=bucket, key_prefix=s3_prefix)

print(f"Uploaded model to {s3_model_path}")


print("Deploying...")
pytorch_model = PyTorchModel(
    model_data=s3_model_path,
    role=role,
    framework_version="2.0.0",
    py_version="py310",
    entry_point="inference.py",
    sagemaker_session=session,
)
print("Deployed!")


print("Creating new predictor...")
predictor = pytorch_model.deploy(
    initial_instance_count=1,
    instance_type=DEPLOY_DEVICE,
    serializer=JSONSerializer(),
    deserializer=JSONDeserializer(),
)


print(f"Endpoint Name: {predictor.endpoint_name}")

# predictor.delete_endpoint()
