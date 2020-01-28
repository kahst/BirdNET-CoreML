import coremltools
import tensorflow as tf
import tfcoreml
import custom_layers

keras_model_path = 'model/BirdNET_1000_RAW_model_with_custom_layer_without_stft.h5'

keras_model = tf.keras.models.load_model(keras_model_path,
                                         custom_objects={'SimpleSpecLayer': custom_layers.SimpleSpecLayer},
                                         compile=False
                                         )

print(keras_model.summary())

print(keras_model.input_shape)


tf.keras.models.save_model(
        keras_model,
        'model/keras_model_without_stft/',
        overwrite=True,
        include_optimizer=False,
        save_format='tf',
        signatures=None,
        options=None
    )

# get input, output node names for the TF graph from the Keras model
input_name = keras_model.inputs[0].name.split(':')[0]
keras_output_node_name = keras_model.outputs[0].name.split(':')[0]
graph_output_node_name = keras_output_node_name.split('/')[-1]

model = tfcoreml.convert(tf_model_path='model/keras_model_without_stft/',
                         input_name_shape_dict={input_name: (1, 384, 257)},
                         output_feature_names=[graph_output_node_name],
                         minimum_ios_deployment_target='13')

model.save('model/BirdNET_1000_RAW_model_with_custom_layer_without_stft.mlmodel')


