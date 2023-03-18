import sagemaker
from sagemaker.s3 import S3Uploader

sagemaker_session = sagemaker.Session()

bucket = sagemaker_session.default_bucket()

prefix = 'iter1'

inputs = S3Uploader.upload("./data/processed", "s3://{}/{}/data".format(bucket, prefix))

print(f'Done uploading to: {inputs}')
