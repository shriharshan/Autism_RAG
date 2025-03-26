import json
from typing import List
import gradio as gr
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline, ChatHuggingFace
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.docstore.document import Document

model = HuggingFacePipeline.from_model_id(
    model_id="HuggingFaceTB/SmolLM2-360M-Instruct",
    task="text-generation",
    pipeline_kwargs=dict(
        max_new_tokens=512,
        do_sample=False,
        repetition_penalty=1.03,
        return_full_text=False,
    ),
)

llm = ChatHuggingFace(llm=model)

def create_embeddings_model() -> HuggingFaceEmbeddings:
    model_name = "BAAI/bge-m3"
    model_kwargs = {
        'device': 'cuda',
        'trust_remote_code': True,
    }
    encode_kwargs = {'normalize_embeddings': True}
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
        show_progress=True
    )

embeddings = create_embeddings_model()

def load_faiss_retriever(path: str) -> FAISS:
    vector_store = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
    return vector_store.as_retriever(search_kwargs={"k": 10})

def load_bm25_retriever(load_path: str) -> BM25Retriever:
    with open(load_path, "r", encoding="utf-8") as f:
        docs_json = json.load(f)
    documents = [Document(page_content=doc["page_content"], metadata=doc["metadata"]) for doc in docs_json]
    return BM25Retriever.from_documents(documents, language="english")

class EmbeddingBM25RerankerRetriever:
    def __init__(self, vector_retriever, bm25_retriever, reranker):
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.reranker = reranker

    def invoke(self, query: str):
        vector_docs = self.vector_retriever.invoke(query)
        bm25_docs = self.bm25_retriever.invoke(query)
        combined_docs = vector_docs + [doc for doc in bm25_docs if doc not in vector_docs]
        return self.reranker.compress_documents(combined_docs, query)

faiss_path = "VectorDB/faiss_index"
bm25_path = "VectorDB/bm25_index.json"

faiss_retriever = load_faiss_retriever(faiss_path)
bm25_retriever = load_bm25_retriever(bm25_path)
reranker_model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3")
reranker = CrossEncoderReranker(top_n=4, model=reranker_model)
retriever = EmbeddingBM25RerankerRetriever(faiss_retriever, bm25_retriever, reranker)

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an AI research assistant specializing in Autism research, powered by a retrieval system of curated PubMed documents.

                Response Guidelines:
                - Provide precise, evidence-based answers drawing directly from retrieved medical research
                - Synthesize information from multiple documents when possible
                - Clearly distinguish between established findings and emerging research
                - Maintain scientific rigor and objectivity

                Query Handling:
                - Prioritize direct, informative responses
                - When document evidence is incomplete, explain the current state of research
                - Highlight areas where more research is needed
                - Never introduce speculation or unsupported claims

                Contextual Integrity:
                - Ensure all statements are traceable to specific research documents
                - Preserve the nuance and complexity of scientific findings
                - Communicate with clarity, avoiding unnecessary medical jargon

                Knowledge Limitations:
                - If no relevant information is found, state: "Current research documents do not provide a comprehensive answer to this specific query."
                """),
    MessagesPlaceholder("chat_history"),
    ("human", "Context:\n{context}\n\nQuestion: {input}")
])

def format_context(docs) -> str:
    return "\n\n".join([f"Doc {i+1}: {doc.page_content}" for i, doc in enumerate(docs)])

def chat_with_rag(query: str, history: List[tuple[str, str]]) -> str:
    chat_history = []
    for human, ai in history:
        chat_history.append(HumanMessage(content=human))
        chat_history.append(AIMessage(content=ai))

    docs = retriever.invoke(query)
    context = format_context(docs)

    prompt_input = {
        "chat_history": chat_history,
        "context": context,
        "input": query
    }
    prompt = qa_prompt.format(**prompt_input)

    response = llm.invoke(prompt)
    return response.content

chat_interface = gr.ChatInterface(
    fn=chat_with_rag,
    title="Autism RAG Chatbot",
    description="Ask questions about Autism.",
    examples=["What causes Autism?", "How is Autism treated?", "What is Autism"],
)

if __name__ == "__main__":
    chat_interface.launch(share=True)
