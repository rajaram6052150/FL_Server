from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    Environment,
    CodeConfiguration,
)

from azure.identity import DefaultAzureCredential


# ==========================================
# AZURE CONFIG
# ==========================================

SUBSCRIPTION_ID = "0746f4dd-3e9d-4b6c-be49-bdcd4149395f"

RESOURCE_GROUP = "flower-rg"

WORKSPACE_NAME = "flower-wsp"

ENDPOINT_NAME = "churn-endpoint"

DEPLOYMENT_NAME = "blue"


# ==========================================
# CONNECT TO AZURE ML
# ==========================================

ml_client = MLClient(
    DefaultAzureCredential(),
    SUBSCRIPTION_ID,
    RESOURCE_GROUP,
    WORKSPACE_NAME,
)

print("Connected to Azure ML!")


# ==========================================
# CREATE ENDPOINT
# ==========================================

endpoint = ManagedOnlineEndpoint(
    name=ENDPOINT_NAME,
    auth_mode="key",
)

print("\nCreating endpoint...")

ml_client.begin_create_or_update(endpoint).result()

print("Endpoint created!")


# ==========================================
# CREATE ENVIRONMENT
# ==========================================

env = Environment(
    name="federated-env",
    conda_file="deployment/environment.yml",
    image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest",
)


# ==========================================
# CREATE DEPLOYMENT
# ==========================================

deployment = ManagedOnlineDeployment(

    name=DEPLOYMENT_NAME,

    endpoint_name=ENDPOINT_NAME,

    model="azureml:federated-churn-model:1",

    environment=env,

    code_configuration=CodeConfiguration(
        code="deployment",
        scoring_script="score.py",
    ),

    instance_type="Standard_DS1_v2",

    instance_count=1,
)

print("\nDeploying model...")

ml_client.begin_create_or_update(
    deployment
).result()

print("\nDEPLOYMENT SUCCESSFUL!")


# ==========================================
# ROUTE TRAFFIC
# ==========================================

endpoint.traffic = {
    "blue": 100
}

ml_client.begin_create_or_update(endpoint).result()

print("\nTraffic assigned successfully!")

print("\nEndpoint Ready!")

