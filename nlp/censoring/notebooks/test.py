import logging
root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)
root_logger.addHandler(logging.StreamHandler())
from asr_dataset.police import BpcETL
from asr_dataset.constants import DataSizeUnit, Cluster

cluster = Cluster['RCC']
root_logger.info('Creating class')
etl = BpcETL(cluster)
root_logger.info('Extracting')
data = etl.extract()
root_logger.info('Transforming')
data = etl.transform(data)
root_logger.info('Ambiguizing')
ambig = etl._resolve_ambiguity(data)
root_logger.info('Describing')
etl.describe(ambig)
