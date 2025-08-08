import psycopg2
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
# Replace deprecated imports with new ones
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import TimescaleVector
from timescale_vector import client, pgvectorizer
from datetime import timedelta
import openai, os
from dotenv import load_dotenv, find_dotenv
import openai

# Load environment variables
_ = load_dotenv(find_dotenv(), override=True)
service_url = "postgres url with creds"
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
openai.api_key = OPENAI_API_KEY

with psycopg2.connect(service_url) as conn:
    with conn.cursor() as cursor:
        # Drop the blog table if it already exists to avoid schema conflicts
        cursor.execute('DROP TABLE IF EXISTS blog')

        # Re-create the blog table with the correct schema
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS blog (
                id              SERIAL PRIMARY KEY,
                title           TEXT NOT NULL,
                author          TEXT NOT NULL,
                contents        TEXT NOT NULL,
                category        TEXT NOT NULL,
                published_time  TIMESTAMPTZ NULL --NULL if not yet published
            );
        ''')

with psycopg2.connect(service_url) as conn:
    with conn.cursor() as cursor:
        cursor.execute('''
            INSERT INTO blog (title, author, contents, category, published_time) VALUES ('First Post', 'Matvey Arye', 'some super interesting content about cats.', 'AI', '2021-01-01'
            );
        ''')
    conn.commit()


def get_document(blog):
    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    docs = []
    for chunk in text_splitter.split_text(blog['contents']):
        content = f"Author {blog['author']}, title: {blog['title']}, contents:{chunk}"
        metadata = {
            "id": str(client.uuid_from_time(blog['published_time'])),
            "blog_id": blog['id'],
            "author": blog['author'],
            "category": blog['category'],
            "published_time": blog['published_time'].isoformat(),
        }
        docs.append(Document(page_content=content, metadata=metadata))
    return docs

with psycopg2.connect(service_url) as conn:
    with conn.cursor() as cursor:
        # Add the embedding column with the correct dimension (1536 for ada embeddings)
        cursor.execute('''
            ALTER TABLE blog_embedding 
            ADD COLUMN IF NOT EXISTS embedding VECTOR(1536);
        ''')
        conn.commit()
def embed_and_write(blog_instances, vectorizer):
    embedding = OpenAIEmbeddings(model="text-embedding-ada-002")
    vector_store = TimescaleVector(
        collection_name="blog_embedding",
        service_url=service_url,
        embedding=embedding,
        time_partition_interval=timedelta(days=30),
    )

    # delete old embeddings for all ids in the work queue. locked_id is a special column that is set to the primary key of the table being
    # embedded. For items that are deleted, it is the only key that is set.
    metadata_for_delete = [{"blog_id": blog['locked_id']} for blog in blog_instances]
    vector_store.delete_by_metadata(metadata_for_delete)

    documents = []
    for blog in blog_instances:
        # skip blogs that are not published yet, or are deleted (in which case the column is NULL)
        if blog['published_time'] != None:
            documents.extend(get_document(blog))

    if len(documents) == 0:
        return

    texts = [d.page_content for d in documents]
    metadatas = [d.metadata for d in documents]
    ids = [d.metadata["id"] for d in documents]
    vector_store.add_texts(texts, metadatas, ids,embedding)


# this job should be run on a schedule
vectorizer = pgvectorizer.Vectorize(service_url, 'blog')
while vectorizer.process(embed_and_write) > 0:
    pass

embedding = OpenAIEmbeddings(model="text-embedding-ada-002")
vector_store = TimescaleVector(
    collection_name="blog_embedding",
    service_url=service_url,
    embedding=embedding,
    time_partition_interval=timedelta(days=30),
)

res = vector_store.similarity_search_with_score("Blogs about cats")
print(res)
