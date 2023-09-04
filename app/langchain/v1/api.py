import logging
import os

import magic
import openai
from fastapi import APIRouter, HTTPException, UploadFile, File, Depends

from app.langchain.v1.config import (
    settings,
    AWS_OPENSEARCH_HOST,
    AWS_OPENSEARCH_USERNAME,
    AWS_OPENSEARCH_PASSWORD,
)
from app.langchain.v1.models import (
    Status,
    HealthCheckResponse,
    DocumentEmbeddingsResponse,
    QueryEmbeddingsRequest,
    QueryEmbeddingsResponse,
    DocumentEmbeddingsRequest,
)

from app.langchain.v1.utils.ias_document_processing import process_pdf_file
from app.langchain.v1.utils.ias_openai_langchain import (
    IASOpenaiEmbeddings,
    IASOpenaiLLM,
    IASOpenaiConversationalLLM,
)
from langchain.vectorstores import OpenSearchVectorSearch
from langchain.chains import (
    RetrievalQA,
    ConversationalRetrievalChain,
)
from langchain.memory import ConversationBufferMemory
# Memory for POC use. A database can be used for better & longer persistence.
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

router = APIRouter()
logger = logging.getLogger()
logger.setLevel(settings.log_level)

openai.api_type = "azure"


@router.get(
    "/health",
    status_code=200,
    tags=["Health check"],
)
async def health_check():
    return HealthCheckResponse(status=Status.success, message="Healthy")


@router.post(
    "/document_embeddings",
    status_code=200,
    tags=["text embeddings"],
    description="Index pdf documents in AWS Opensearch. New files support coming",
    response_model=DocumentEmbeddingsResponse,
)
async def langchain_document_embeddings(
    request: DocumentEmbeddingsRequest = Depends(), file: UploadFile = File(...)
):
    try:
        req = request.dict()
        # Create index name based on user's input just for now. This will change as is not validated.
        input = req["index_name"]
        input = input.replace(" ", "_")
        index_name = f"lang_{input}"
        # Create vector db from AWS OpenSearch
        vector_db = OpenSearchVectorSearch(
            index_name=index_name,
            embedding_function=IASOpenaiEmbeddings(engine=req["engine"]),
            opensearch_url=AWS_OPENSEARCH_HOST,
            http_auth=(AWS_OPENSEARCH_USERNAME, AWS_OPENSEARCH_PASSWORD),
            is_aoss=False,
        )

        # Read file
        contents = await file.read()

        # Determine file type using magic library
        file_type = magic.from_buffer(contents, mime=True)

        if file_type == "application/pdf":
            # File is PDF. Use PDF loader
            logger.info("PDF uploaded")
            page_content, metadata = await process_pdf_file(file)
        elif (
            file_type
            == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        ):
            # File is a DOCX
            # Use DOCX loader
            logger.warning("docx uploaded")
        else:
            logger.warning("other file uploaded")

        vectors = vector_db.add_texts(texts=page_content, metadatas=metadata)
    except Exception as e:
        logger.error("langchain document embeddings error")
        logger.error(e.status_code)
        logger.error(str(e))
        raise HTTPException(status_code=e.status_code, detail=f"Error: {str(e)}")
    return DocumentEmbeddingsResponse(
        status=Status.success, index_name=index_name, embeddings_ids=vectors
    )


@router.post(
    "/query_embeddings",
    status_code=200,
    tags=["text embeddings"],
    description="Given an index name and a query it will return the best answer by using langchain RetrievalQA",
    response_model=QueryEmbeddingsResponse,
)
async def langchain_query_embeddings(request: QueryEmbeddingsRequest):
    try:
        vector_db = OpenSearchVectorSearch(
            index_name=request.index_name,
            embedding_function=IASOpenaiEmbeddings(engine=request.embeddings_engine),
            opensearch_url=AWS_OPENSEARCH_HOST,
            http_auth=(AWS_OPENSEARCH_USERNAME, AWS_OPENSEARCH_PASSWORD),
            is_aoss=False,
        )
        qa = RetrievalQA.from_chain_type(
            IASOpenaiLLM(
                engine=request.completion_engine,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            ),
            chain_type="stuff",
            retriever=vector_db.as_retriever(),
            
        )
        answer = qa.run(request.query)
        

    except Exception as e:
        logger.error("langchain query embeddings error")
        logger.error(e.status_code)
        logger.error(str(e))
        raise HTTPException(status_code=e.status_code, detail=f"Error: {str(e)}")
    return QueryEmbeddingsResponse(status=Status.success, answer=answer)


@router.post(
    "/query_embeddings_conversational",
    status_code=200,
    tags=["text embeddings"],
    description="Given an index name and a query it will return the best answer by using langchain Conversational QA",
    response_model=QueryEmbeddingsResponse,
)
async def langchain_query_embeddings(request: QueryEmbeddingsRequest):
    chat_history =[]
    try:
        vector_db = OpenSearchVectorSearch(
            index_name=request.index_name,
            embedding_function=IASOpenaiEmbeddings(engine=request.embeddings_engine),
            opensearch_url=AWS_OPENSEARCH_HOST,
            http_auth=(AWS_OPENSEARCH_USERNAME, AWS_OPENSEARCH_PASSWORD),
            is_aoss=False,
        )
        qa = ConversationalRetrievalChain.from_llm(
            IASOpenaiConversationalLLM(
                engine=request.completion_engine,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            ),
            chain_type="stuff",
            retriever=vector_db.as_retriever(),
            return_source_documents=True,
        )
        result = qa({"question": request.query,"chat_history":chat_history})
        metadata_list = []
        chat_history.append((request.query, result['answer']))
        c = 1
        for doc in result['source_documents']:
            metadata = doc.metadata
            metadata_list.append({os.path.basename(metadata['source']): doc.page_content})
            c += 1
        result['metadata']=metadata_list

    except Exception as e:
        logger.error("langchain query embeddings error")
        logger.error(e.status_code)
        logger.error(str(e))
        raise HTTPException(status_code=e.status_code, detail=f"Error: {str(e)}")
    return QueryEmbeddingsResponse(status=Status.success, answer=result["answer"],citation =result['metadata'])
