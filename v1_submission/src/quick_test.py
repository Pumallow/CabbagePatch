# Quick diagnostic
from llm import get_vectorstore

vs = get_vectorstore()
print("Total documents:", vs._collection.count())

# Test retrieval
retriever = vs.as_retriever(search_kwargs={"k": 3})
docs = retriever.invoke("How many Champions League titles has Ronaldo won?")
for d in docs:
    print(d.page_content)