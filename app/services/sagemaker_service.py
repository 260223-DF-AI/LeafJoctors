"""
FastAPI Router that handles functionality
related to SageMaker & our leaf vision model
"""

import json

import boto3
from fastapi import UploadFile, File, HTTPException

# vision_router = APIRouter(prefix="/vision", tags=["vision"])

# Create SageMaker runtime client
sagemaker_runtime = boto3.client("sagemaker-runtime", region_name="us-east-1")

ENDPOINT_NAME = "your-sagemaker-endpoint-name"


# @vision_router.post("/")
async def predict(file: UploadFile = File(...)):
    try:
        # Read image bytes
        image_bytes = await file.read()

        # Invoke SageMaker endpoint
        response = sagemaker_runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType=file.content_type,  # e.g. image/jpeg
            Body=image_bytes
        )

        # Read response body
        result = response["Body"].read().decode("utf-8")

        # If your model returns JSON
        try:
            result_json = json.loads(result)
            return {"prediction": result_json}
        except:
            return {"prediction": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
