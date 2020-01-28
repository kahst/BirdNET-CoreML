import coremltools
import tensorflow as tf
import tfcoreml

keras_model_path = 'model/BirdNET_1000_RAW_model_without_preprocessing_custom_layer.h5'

keras_model = tf.keras.models.load_model(keras_model_path)

print(keras_model.summary())

print(keras_model.input_shape)

# get input, output node names for the TF graph from the Keras model
input_name = keras_model.inputs[0].name.split(':')[0]
keras_output_node_name = keras_model.outputs[0].name.split(':')[0]
graph_output_node_name = keras_output_node_name.split('/')[-1]

model = tfcoreml.convert(tf_model_path=keras_model_path,
                         input_name_shape_dict={input_name: (1, 257, 384, 1)},
                         output_feature_names=[graph_output_node_name],
                         minimum_ios_deployment_target='13')
model.save('model/BirdNET_1000_RAW_model_without_preprocessing_custom_layer.mlmodel')


