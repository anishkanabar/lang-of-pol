import sys
import tensorflow as tf

print("Num CPUs Available: ", len(tf.config.list_physical_devices('CPU')))
#print("CPUs Available: ", tf.config.list_physical_devices('CPU'), file=sys.stderr)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
#print("GPUs Available: ", tf.config.list_physical_devices('GPU'), file=sys.stderr)

