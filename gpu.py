import tensorflow as tf

# List available physical devices
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available:", len(physical_devices))
print("Physical GPUs:", physical_devices)

if len(physical_devices) > 0:
    # Enable memory growth
    try:
        for gpu in physical_devices:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print("Logical GPUs:", logical_gpus)
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
