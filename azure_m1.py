from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model
from azure.identity import DefaultAzureCredential


SUBSCRIPTION_ID = "0746f4dd-3e9d-4b6c-be49-bdcd4149395f"

RESOURCE_GROUP = "flower-rg"

WORKSPACE_NAME = "flower-wsp"


credential = DefaultAzureCredential()

ml_client = MLClient(
    credential=credential,
    subscription_id=SUBSCRIPTION_ID,
    resource_group_name=RESOURCE_GROUP,
    workspace_name=WORKSPACE_NAME,
)


def upload_model_to_azure():

    model = Model(
        path="models/final_model.pth",
        name="telco-fl-model",
        description="Federated Learning Churn Model",
        type="custom_model",
    )

    registered_model = ml_client.models.create_or_update(model)

    print("\nMODEL REGISTERED TO AZURE ML!")

    print(f"Name: {registered_model.name}")

    print(f"Version: {registered_model.version}")