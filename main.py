# Import libraries
from sqlalchemy import create_engine, Column, String, LargeBinary
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy_utils import database_exists, create_database
import shopify
import pandas as pd
import pinecone
from pinecone_text.sparse import BM25Encoder
from tqdm.auto import tqdm
from PIL import Image
import requests
import numpy as np
from fastapi import FastAPI, Path
from sentence_transformers import SentenceTransformer
import torch
from typing import Optional
from pydantic import BaseModel
import pickle

### Constants ###

# Default Shopify constants
# Used for testing
SHOPIFY_SHOP_HANDLE = 'pinecone-testing'
SHOPIFY_API_KEY = '<SHOPIFY_API_KEY>'
SHOPIFY_PASSWORD = '<SHOPIFY_PASSWORD>'
SHOPIFY_API_VERSION = '2023-07'

# PostgreSQL constants
PGSQL = {
    "pguser":"postgres",
    "pgpassword":"postgres",
    "pghost":"localhost",
    "pgport":5432,
    "pgdb":"shopify"
}

# Pinecone Constants
PINECONE_API_KEY = '0b4ba524-c947-4317-8642-d63265789c04'
PINECONE_ENV='gcp-starter'
INDEX_PREFIX = 'hybrid-search-'
BATCH_SIZE = 200

### Configurations ###
# SQL Alchemy model configurations
Base = declarative_base()

# Pinecone configurations
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

### SQLALchemy Models ###
# Product Model
class Product(Base):
    __tablename__="products"
    id = Column("id", String, primary_key=True)
    shop_id = Column("shop_id", String)
    title = Column("title", String)
    vendor = Column("vendor", String)
    created_at=Column("created_at", String)
    updated_at = Column("updated_at", String)
    status = Column("status", String)
    image = Column("image", String)

    def __init__(self,id,shop_id, title,vendor, created_at, updated_at, status, image):
        self.id = id
        self.shop_id = shop_id
        self.title = title
        self.vendor =vendor
        self.created_at = created_at
        self.updated_at = updated_at
        self.status = status
        self.image = image

# BM25Model Model
class BM25Model(Base):
    __tablename__="bm25_models"
    id = Column("id", String, primary_key=True)
    model = Column("model", LargeBinary)

    def __init__(self, id, model):
        self.id = id
        self.model = model

### FastAPI Models ###
# Shop Model
class ShopModel(BaseModel):
    shopify_shop_handle: str
    shopify_api_key: str
    shopify_api_version: str
    shopify_password: str

### Utility functions ###
# Get PostgreSQL engine for saving data
def get_engine(user, password, host, port, db):
    url = f"postgresql://{user}:{password}@{host}:{port}/{db}"
    if not database_exists(url):
        create_database(url)
    engine = create_engine(url, pool_size=100, echo=False)
    return engine

# Get SQLAlchemy session with PostgreSQL
def get_session(engine):
    Base.metadata.create_all(bind=engine)
    session = sessionmaker(bind=engine)
    return session

# Get relevant data from shopify
def get_data(object_name):
    all_data = []
    attribute = getattr(shopify, object_name)
    data = attribute.find(since_id=0, limit=250)
    for d in data:
        all_data.append(d)
    while data.has_next_page():
        data = data.next_page()
        for d in data:
            all_data.append(d)
    return all_data

# Access shopify store and save product data in PostgreSQL
def save_shopify_data(engine, shopify_shop_handle=SHOPIFY_SHOP_HANDLE, shopify_api_key=SHOPIFY_API_KEY, shopify_password=SHOPIFY_PASSWORD, shopify_api_version=SHOPIFY_API_VERSION):
    shop_url = "https://{}.myshopify.com/admin/api/{}".format(shopify_shop_handle, shopify_api_version)
    shopify.ShopifyResource.set_site(shop_url)
    shopify.ShopifyResource.set_user(shopify_api_key)
    shopify.ShopifyResource.set_password(shopify_password)

    # Create SQLAlchemy session
    Session = get_session(engine)
    session = Session()

    # Saving product data
    for data in get_data('Product'):
        # Print data
        # print(data.title)

        # Since there can be products without images
        id = str(data.id)
        shop_id = shopify_shop_handle
        title = data.title
        vendor = data.vendor
        created_at = data.created_at
        updated_at = data.updated_at
        status = data.status
        image = data.image
        if  image is None:
            image = ''
        else:
            image = data.image.src

        checkProduct = session.query(Product).filter(Product.id == id)
        session.commit()
        if checkProduct.first() is None:
            product = Product(id, shop_id=shop_id, title=title, vendor=vendor, created_at=created_at, updated_at=updated_at, status=status, image=image)
            # Save products in db
            session.add(product)
            session.commit()
            print(f"Product {id}:{title} added.")
        else:
            print(f"Product {id}:{title} already exists.")
    session.close()

# Get dataframe from table
def get_dataframe_from_db(engine, shop_id):
    df = pd.read_sql_query(f"select * from products where shop_id='{shop_id}'",con=engine)
    return df

# Create pinecone index
def create_pinecone_index(index_name):
    if index_name not in pinecone.list_indexes():
        # create the index
        pinecone.create_index(index_name, dimension=512, metric="dotproduct", pod_type="s1")

# Connect to a newly generated pinecone index
def connect_pinecone_index(shop_id):
    # Create pinecone index
    index_id = INDEX_PREFIX + shop_id
    create_pinecone_index(index_id)
    # Connect to pinecone index
    index = pinecone.Index(index_id)
    return index

# Convert image urls to Image Objects
def convert_to_img(url):
  if str(url).lower() == 'nan' or str(url).strip() == '':
    return np.nan

  img = Image.open(requests.get(url, stream=True).raw)
  return img

# Clean the CSV from DB
def clean_csv(engine, shop_id):
    # Uncleaned dataset
    metadata = get_dataframe_from_db(engine, shop_id)
    # Cleaning dataset
    metadata['image_file'] = metadata['image'].apply(convert_to_img)
    metadata = metadata.dropna(subset=['id','title','image', 'image_file']).reset_index(drop=True)
    images = metadata['image_file']
    metadata = metadata.drop(columns=['image_file'])
    return (metadata, images)

# Upsert documents to pinecone
def upsert_documents(metadata, images, index, model):
    # Create bm25 model
    bm25 = BM25Encoder()
    # Fit the model
    bm25.fit(metadata['title'])
    # Define number of rows in the df
    num_rows = len(metadata.index)
    # Run upserting
    for i in tqdm(range(0, num_rows, BATCH_SIZE)):
        #find end of batch
        i_end = min(i+BATCH_SIZE, num_rows)
        #extract metadata batch
        meta_batch = metadata.iloc[i:i_end]
        meta_dict = meta_batch.to_dict(orient="records")
        # concatinate all metadata field except for id, created_at, updated_at to form a single string
        meta_batch = [" ".join(x) for x in meta_batch.loc[:, ~meta_batch.columns.isin(['id', 'year','created_at','updated_at','image'])].values.tolist()]
        # extract image batch
        img_batch = images[i:i_end]
        # create sparse BM25 vectors
        sparse_embeds = bm25.encode_documents([text for text in meta_batch])
        # create dense vectors
        dense_embeds = model.encode(img_batch).tolist()
        # create unique IDs
        ids = [str(x) for x in range(i, i_end)]

        upserts = []
        # loop through the data and create dictionaries for uploading documents to pinecone index
        for _id, sparse, dense, meta in zip(ids, sparse_embeds, dense_embeds, meta_dict):
            upserts.append({'id': _id,'sparse_values': sparse,'values': dense,'metadata': meta})
        # upload the documents to the new hybrid index
        index.upsert(upserts)
    return bm25

# Save bm25 model
def save_bm25_model(bm25model, shop_id):
    # Create pickle byte object
    model_bytes = pickle.dumps(bm25model)
    # Create database id
    index_id = INDEX_PREFIX + shop_id
    # Create SQLAlchemy session
    Session = get_session(engine)
    session = Session()
    # Delete previous model
    checkModel = session.query(BM25Model).filter(BM25Model.id == index_id)
    session.commit()
    if checkModel.first() is not None:
        checkModel.delete()
    # Save model
    bm25Model = BM25Model(id=index_id, model=model_bytes)
    # Save products in db
    session.add(bm25Model)
    session.commit()
    session.close()

# Load bm25 model
def load_bm25_model(shop_id):
    # Create database id
    index_id = INDEX_PREFIX + shop_id
    # Create SQLAlchemy session
    Session = get_session(engine)
    session = Session()
    # Get the relevant model
    model_bytes = session.query(BM25Model).filter(BM25Model.id == index_id).first()
    # Load bm25 model
    bm25model = pickle.loads(model_bytes.model)
    return bm25model

# Search the store with a query
def search_with_query(query, index, bm25, model):
    # create sparse and dense vectors
    sparse = bm25.encode_queries(query)
    dense = model.encode(query).tolist()
    # search
    result = index.query(
        top_k=14,
        vector=dense,
        sparse_vector=sparse,
        include_metadata=True
    )
    return result

### Variables ###
# PostgreSQL engine
engine = get_engine(PGSQL["pguser"], PGSQL["pgpassword"], PGSQL["pghost"], PGSQL["pgport"], PGSQL["pgdb"])

# Create fastapi object
app = FastAPI()

# Check device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# load a CLIP model from huggingface
model = SentenceTransformer(
    'sentence-transformers/clip-ViT-B-32',
    device=device
)

### APIs ###
# /api/customer
@app.post('/api/customer')
def create_pinecone_model(shop: ShopModel):
    # Save shopify data in database
    shopify_shop_handle = shop.shopify_shop_handle
    shopify_api_key = shop.shopify_api_key
    shopify_password = shop.shopify_password
    shopify_api_version = shop.shopify_api_version
    save_shopify_data(engine, shopify_shop_handle, shopify_api_key, shopify_password, shopify_api_version)
    # Create new pinecone index
    index = connect_pinecone_index(shopify_shop_handle)
    # Load the saved data from db
    (metadata, images) = clean_csv(engine, shopify_shop_handle)
    # Upsert documents
    bm25 = upsert_documents(metadata=metadata, images=images, index=index, model=model)
    # Save bm25 model
    save_bm25_model(bm25,shopify_shop_handle)
    # Return result
    return {"shop_id":shopify_shop_handle}

# /api/query=?{query}
@app.get('/api/query')
def get_pinecone_results(query: Optional[str] = None, shopify_shop_handle:Optional[str]=SHOPIFY_SHOP_HANDLE):
    if query == None:
        return {"result":[]}
    else:
        # Connect to pinecone index
        index = connect_pinecone_index(shopify_shop_handle)
        # Load saved model
        bm25 = load_bm25_model(shopify_shop_handle)
        # Get the results
        result = search_with_query(query=query, index=index, bm25=bm25, model=model)
        # Get all the products only
        products = [{"id":product["metadata"]["id"], "title":product["metadata"]["title"], "image":product["metadata"]["image"],"status":product["metadata"]["status"]} for product in result['matches']]
        count  = len(products)
        # Return results
        return {"count": count, "products": products}





