from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer

endpoint_name: str = "" # find from sagemaker "Endpoints" or print statement after deployed
predictor = Predictor(
    endpoint_name=endpoint_name,
    serialize=JSONSerializer(),
    deserialize=JSONDeserializer(),
)


def make_prediction(inp):
    """sample use of prediction"""
    response = predictor.predict(inp)


