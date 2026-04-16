import os
import tarfile

import boto3
import sagemaker
from dotenv import load_dotenv
from sagemaker.deserializers import JSONDeserializer
from sagemaker.pytorch import PyTorchModel
from sagemaker.serializers import IdentitySerializer

load_dotenv()


print("Initial Setup...")
DEPLOY_DEVICE = "ml.m5.large"
TAR_NAME = "model.tar.gz"
LOCAL_MODEL_DIR = "model"

ARN = os.getenv("IAM")
AWS_REGION = (
    os.getenv("SAGEMAKER_REGION")
    or os.getenv("AWS_REGION")
    or os.getenv("AWS_DEFAULT_REGION")
)

if not AWS_REGION:
    raise RuntimeError(
        "Missing AWS region. Set SAGEMAKER_REGION, AWS_REGION, or AWS_DEFAULT_REGION "
        "to a SageMaker-supported region such as us-east-1 before running deploy.py."
    )

print(f"Deploying on {DEPLOY_DEVICE}")
print(f"Using AWS region: {AWS_REGION}")

# save model to a directory
print("Saving the model...")
os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)  # prep the path for the model
model_path = os.path.join(LOCAL_MODEL_DIR, "drfrond.pth")
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Missing model weights at {model_path}")

code_dir = os.path.join(LOCAL_MODEL_DIR, "code")  # prep the path for the package
os.makedirs(code_dir, exist_ok=True)

if not os.path.exists("./model.tar.gz"):
    with tarfile.open(TAR_NAME, "w:gz") as tar:  # create the archive
        tar.add(model_path, arcname="model.pth")

    print(f"Saved model to {TAR_NAME}")

else:
    print("Model zip already exists.")


print("Uploading to S3...")
try:
    boto_session = boto3.Session(region_name=AWS_REGION)
    session = sagemaker.Session(boto_session=boto_session)
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
    source_dir=code_dir,
    sagemaker_session=session,
)
print("Deployed!")


print("Creating new predictor...")
predictor = pytorch_model.deploy(
    initial_instance_count=1,
    instance_type=DEPLOY_DEVICE,
    serializer=IdentitySerializer(content_type="application/octet-stream"),
    deserializer=JSONDeserializer(),
)


print(f"Endpoint Name: {predictor.endpoint_name}")

# predictor.delete_endpoint()
