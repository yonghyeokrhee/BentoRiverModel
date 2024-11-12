import bentoml
from typing import Any
from pydantic import BaseModel

class InputFeatures(BaseModel):
    ordinal_date: int = 736489
    gallup: float = 37.843213
    ipsos: float = 38.07067899999999
    morning_consult: float = 42.318749
    rasmussen: float = 40.104692
    you_gov: float = 38.636914000000004

@bentoml.service(
    resources={"cpu": "2"},
    traffic={"timeout": 10},
)
class RiverOnlineLearner:
    bento_model = bentoml.mlflow.get("river_arf_model:latest")

    def __init__(self):
        self.model = bentoml.mlflow.load_model(self.bento_model)

    @bentoml.api(input_spec=InputFeatures)
    def predict(self, **params: Any):
        return self.model.predict(params)
