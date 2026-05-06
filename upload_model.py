# upload_model.py

"""
Upload Final Federated Learning Model to Azure ML Registry
"""

from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model
from azure.identity import DefaultAzureCredential

# =========================================================
# AZURE ML CONFIG
# =========================================================

SUBSCRIPTION_ID = "0746f4dd-3e9d-4b6c-be49-bdcd4149395f"

RESOURCE_GROUP = "flower-rg"

WORKSPACE_NAME = "flower-wsp"

# =========================================================
# CONNECT TO AZURE ML
# =========================================================

print("=" * 60)
print("Connecting to Azure ML Workspace...")
print("=" * 60)

ml_client = MLClient(
    credential=DefaultAzureCredential(),
    subscription_id=SUBSCRIPTION_ID,
    resource_group_name=RESOURCE_GROUP,
    workspace_name=WORKSPACE_NAME,
)

print("Connected successfully!")

# =========================================================
# MODEL PATH
# =========================================================

MODEL_PATH = "models/final_model.pth"

# =========================================================
# CREATE MODEL OBJECT
# =========================================================

model = Model(
    path=MODEL_PATH,
    name="federated-churn-model",
    description="Final global model from Flower Federated Learning",
    type="custom_model",
)

# =========================================================
# UPLOAD MODEL
# =========================================================

print("\nUploading model to Azure ML Registry...")

registered_model = ml_client.models.create_or_update(model)

print("\n" + "=" * 60)
print("MODEL UPLOADED SUCCESSFULLY!")
print("=" * 60)

print(f"Model Name    : {registered_model.name}")
print(f"Model Version : {registered_model.version}")
print(f"Model ID      : {registered_model.id}")

print("=" * 60)