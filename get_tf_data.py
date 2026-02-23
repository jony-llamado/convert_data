import tensorflow_datasets as tfds
import tensorflow as tf
import os

# Load from GCS
b = tfds.builder_from_directory(
    builder_dir='gs://gresearch/robotics/fractal20220817_data/0.1.0'
)
ds = b.as_dataset(split='train[:10]')

# Save locally as TFRecord files
local_path = 'rlds_data'
os.makedirs(local_path, exist_ok=True)

tf.data.experimental.save(ds, local_path)

# Save the element spec so you can reload it later
spec = ds.element_spec
tf.saved_model.save(tf.Module(), local_path + '/spec')