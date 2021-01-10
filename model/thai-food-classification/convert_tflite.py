import tensorflow as tf

#%%
# Import file name manully !!
saved_model = 'saved_fine-tuned-model/saved-fine-model-20-0.94.hdf5'
loaded_model = tf.keras.models.load_model(saved_model, compile=False)

#%%
from tensorflow.keras.optimizers import SGD

loaded_model.compile(
    optimizer=SGD(lr=0.00001, momentum=0.9),
    loss='categorical_crossentropy',
    metrics=['accuracy']
    )

#%%
TFLITE_MODEL = 'tflite_models/tflite_model.tflite'
TFLITE_QUANT_MODEL = 'tflite_models/tflite_quant.tflite'

#%%
# Get the concrete function from the Keras model.
run_model = tf.function(lambda x : loaded_model(x))

# Save the concrete function.
concrete_func = run_model.get_concrete_function(
    tf.TensorSpec(loaded_model.inputs[0].shape, loaded_model.inputs[0].dtype)
)

# Convert the model
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
converted_tflite_model = converter.convert()
open(TFLITE_MODEL, "wb").write(converted_tflite_model)

#%%
# Convert the model to quantized version with post-training quantization
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_quant_model = converter.convert()
open(TFLITE_QUANT_MODEL, "wb").write(tflite_quant_model)