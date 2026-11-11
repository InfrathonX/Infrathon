"""Controller-only connection to the authenticated platform artifact bucket."""
import os
from urllib.parse import urlsplit


def storage_connection():
    from pyromind_sdk import PyroMindAPIClient
    from minio import Minio
    import urllib3
    info = PyroMindAPIClient(timeout=30, max_retries=0).profile.get_storage_info()
    endpoint = urlsplit(info.url if '://' in info.url else 'https://' + info.url)
    client = Minio(endpoint.netloc, access_key=info.access_key,
        secret_key=info.secret_key, region='us-east-1', secure=endpoint.scheme == 'https',
        http_client=urllib3.PoolManager(timeout=urllib3.Timeout(connect=10, read=60),
            retries=urllib3.Retry(total=1, connect=1, read=0, status=0), cert_reqs='CERT_REQUIRED'))
    bucket = os.getenv('PYROMIND_STORAGE_BUCKET')
    if not bucket:
        buckets = client.list_buckets()
        if len(buckets) != 1:
            raise ValueError('Set PYROMIND_STORAGE_BUCKET to select the artifact bucket')
        bucket = buckets[0].name
    return client, bucket, f'{endpoint.scheme}://{endpoint.netloc}'


def storage_client():
    client, bucket, _ = storage_connection()
    return client, bucket
