import os
import logging
import subprocess
import tensorflow as tf

logger = logging.getLogger('env')

num_cpu = len(tf.config.list_physical_devices('CPU'))
num_gpu = len(tf.config.list_physical_devices('GPU'))
cuda_devs = os.environ['CUDA_VISIBLE_DEVICES']
hostname = subprocess.run("hostname").stdout
logger.info(f"Running on host: {hostname}")
logger.info(f"Num CPUs Available: {num_cpu}")
logger.info(f"Num GPUs Available: {num_gpu}")
logger.info(f"CUDA visible devices: {cuda_devs}")

