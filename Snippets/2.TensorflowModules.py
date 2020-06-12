# TensorFlowDataValidation
"""import tensorflow_data_validation as tfdv
trainStats = tfdv.generate_statistics_from_dataframe(dataframe=dataset)
schema = tfdv.infer_schema(statistics=trainStats)
tfdv.display_schema(schema)

testStats = tfdv.generate_statistics_from_dataframe(dataframe=testingSet)
anomalies = tfdv.validate_statistics(statistics=testStats, schema=schema)

tfdv.display_anomalies(anomalies)

schema.default_environment.append("TRAINING")
schema.default_environment.append("SERVING")

tfdv.get_feature(schema, "---").not_in_environment.append("SERVING")
serving_env_anomalies = tfdv.validate_statistics(testStats, schema, environment="SERVING")"""

# TensorBoard
"""from tensorflow.keras.callbacks import TensorBoard
tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

tensorboard --logdir=logs/."""

# TensorFlowServing
"""from tensorflow.saved_model import simple_save
from tensorflow.keras.backend import get_session
simple_save(
    get_session(),
    export_dir="model/1",
    inputs={"input_image": model.input},
    outputs={t.name: t for t in model.outputs},
)"""

# TensorFlowLite
"""from tensorflow.lite.TFLiteConverter import from_keras_model_file
from tensorflow.lite import Interpreter

with open("Model.tflite", "wb") as f:
    f.write(from_keras_model(model).convert())

interpreter = Interpreter(model_path="Model.tflite")"""

# TensorFlowDistributed
"""from tensorflow.distribute import MirroredStrategy
distribute = MirroredStrategy()

with distribute.scope():
    # Define Model Here"""
