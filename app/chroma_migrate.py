import chromadb
  
client = chromadb.CloudClient(
  api_key='ck-24qfKWesKdXD76czEPe5s8vBFpBQzDKexzsjjx9kqmrh',
  tenant='e68335e5-7a51-4fbe-8d4a-1d15494b467d',
  database='keyword_index'
)

clientP = chromadb.PersistentClient(path="./podak")

col = client.get_or_create_collection(
    name = 'podak'
)

colP = clientP.get_collection(
    name = 'podak',
)

results = colP.get(
        include=["documents", "embeddings", "metadatas"]
    )

print(len(results["embeddings"]))

metadatas = [{"keyword" : m["keyword"]} for m in results["metadatas"]]
col.add(
    ids = results["ids"][:200],
    documents = results["documents"][:200],
    metadatas = metadatas[:200]
)

print('Phase 1 done')
col.add(
    ids = results["ids"][200:],
    documents = results["documents"][200:],
    metadatas = metadatas[200:]
)

print('DONE')
