"""Create an empty collection without embedding calls or overwriting existing data."""
import os
from pathlib import Path

from dotenv import load_dotenv
from pymilvus import MilvusClient

from hybrid_search import COLLECTION_NAME


def main():
    load_dotenv(Path(__file__).with_name('.env'), override=True)
    dimension = int(os.getenv('EMBEDDING_DIMENSIONS') or '1152')
    client = MilvusClient(
        uri=os.getenv('MILVUS_URL') or 'http://127.0.0.1:19530',
        token=os.getenv('MILVUS_TOKEN') or '',
    )
    try:
        if client.has_collection(collection_name=COLLECTION_NAME):
            info = client.describe_collection(collection_name=COLLECTION_NAME)
            field = next(f for f in info['fields'] if f['name'] == 'vector')
            if int(field['params']['dim']) != dimension:
                raise ValueError('Existing collection dimension differs; no data was changed.')
            print('Collection exists; no data was changed:', COLLECTION_NAME)
        else:
            client.create_collection(
                collection_name=COLLECTION_NAME,
                dimension=dimension,
                primary_field_name='id',
                auto_id=False,
                vector_field_name='vector',
                metric_type='COSINE',
                enable_dynamic_field=True,
                consistency_level='Strong',
            )
            print('Created:', COLLECTION_NAME)
    finally:
        client.close()


if __name__ == '__main__':
    main()
