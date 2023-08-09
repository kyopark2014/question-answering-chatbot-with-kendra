import boto3
import json
import datetime
import sys
import os
import time
import PyPDF2
import csv
from io import BytesIO

from langchain.llms.bedrock import Bedrock
from langchain import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import CSVLoader
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.retrievers import AmazonKendraRetriever

module_path = "."
sys.path.append(os.path.abspath(module_path))
from utils import bedrock, print_ww

s3 = boto3.client('s3')
s3_bucket = os.environ.get('s3_bucket') # bucket name
s3_prefix = os.environ.get('s3_prefix')
callLogTableName = os.environ.get('callLogTableName')
configTableName = os.environ.get('configTableName')
endpoint_url = os.environ.get('endpoint_url')
bedrock_region = os.environ.get('bedrock_region')
kendraIndex = os.environ.get('kendraIndex')
roleArn = os.environ.get('roleArn')
modelId = os.environ.get('model_id')
print('model_id: ', modelId)
accessType = os.environ.get('accessType')

# Bedrock Contiguration
bedrock_region = bedrock_region
bedrock_config = {
    "region_name":bedrock_region,
    "endpoint_url":endpoint_url
}
    
# supported llm list from bedrock
if accessType=='aws':  # internal user of aws
    boto3_bedrock = bedrock.get_bedrock_client(
        region=bedrock_config["region_name"],
        url_override=bedrock_config["endpoint_url"])
else: # preview user
    boto3_bedrock = bedrock.get_bedrock_client(
        region=bedrock_config["region_name"])
    
modelInfo = boto3_bedrock.list_foundation_models()    
print('models: ', modelInfo)

parameters = {
    "maxTokenCount":4096, 
    "stopSequences":[],
    "temperature":0,
    "topP":0.9
}

llm = Bedrock(model_id=modelId, client=boto3_bedrock, model_kwargs=parameters)

retriever = AmazonKendraRetriever(index_id=kendraIndex)

def save_configuration(userId, modelId):
    item = {
        'user-id': {'S':userId},
        'model-id': {'S':modelId}
    }

    client = boto3.client('dynamodb')
    try:
        resp =  client.put_item(TableName=configTableName, Item=item)
        print('resp, ', resp)
    except: 
        raise Exception ("Not able to write into dynamodb")            

def load_configuration(userId):
    print('configTableName: ', configTableName)
    print('userId: ', userId)

    client = boto3.client('dynamodb')    
    try:
        key = {
            'user-id': {'S':userId}
        }

        resp = client.get_item(TableName=configTableName, Key=key)
        print('model-id: ', resp['Item']['model-id']['S'])

        return resp['Item']['model-id']['S']
    except: 
        print('No record of configuration!')
        modelId = os.environ.get('model_id')
        save_configuration(userId, modelId)

        return modelId

# store document into Kendra
def store_document(s3_file_name, requestId):
    documentInfo = {
        "S3Path": {
            "Bucket": s3_bucket,
            "Key": s3_prefix+'/'+s3_file_name
        },
        "Title": s3_file_name,
        "Id": requestId
    }

    documents = [
        documentInfo
    ]

    kendra = boto3.client("kendra")
    result = kendra.batch_put_document(
        Documents = documents,
        IndexId = kendraIndex,
        RoleArn = roleArn
    )
    print(result)

# load documents from s3
def load_document(file_type, s3_file_name):
    s3r = boto3.resource("s3")
    doc = s3r.Object(s3_bucket, s3_prefix+'/'+s3_file_name)
    
    if file_type == 'pdf':
        contents = doc.get()['Body'].read()
        reader = PyPDF2.PdfReader(BytesIO(contents))
        
        raw_text = []
        for page in reader.pages:
            raw_text.append(page.extract_text())
        contents = '\n'.join(raw_text)    
        
    elif file_type == 'txt':        
        contents = doc.get()['Body'].read()
    elif file_type == 'csv':        
        body = doc.get()['Body'].read()
        reader = csv.reader(body)        
        contents = CSVLoader(reader)
    
    print('contents: ', contents)
    new_contents = str(contents).replace("\n"," ") 
    print('length: ', len(new_contents))

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=100)
    texts = text_splitter.split_text(new_contents) 
    print('texts[0]: ', texts[0])
        
    docs = [
        Document(
            page_content=t
        ) for t in texts[:3]
    ]
    return docs

def summerize_text(text):
    docs = [
        Document(
            page_content=text
        )
    ]
    prompt_template = """Write a concise summary of the following:

    {text}
                
    CONCISE SUMMARY """

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
    chain = load_summarize_chain(llm, chain_type="stuff", prompt=PROMPT)
    summary = chain.run(docs)
    print('summarized text: ', summary)

    return summary
              
def get_reference(docs):
    reference = "\n\nFrom\n"
    for doc in docs:
        name = doc.metadata['title']
        page = doc.metadata['document_attributes']['_excerpt_page_number']
    
        reference = reference + (str(page)+'page in '+name+'\n')
    return reference

def get_answer_using_template(query):
    relevant_documents = retriever.get_relevant_documents(query)
    print('length of relevant_documents: ', len(relevant_documents))

    if(len(relevant_documents)==0):
        return llm(query)
    else:
        print(f'{len(relevant_documents)} documents are fetched which are relevant to the query.')
        print('----')
        for i, rel_doc in enumerate(relevant_documents):
            print_ww(f'## Document {i+1}: {rel_doc.page_content}.......')
            print('---')

        prompt_template = """Human: Use the following pieces of context to provide a concise answer to the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

        {context}

        Question: {question}
        Assistant:"""
        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )

        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
        result = qa({"query": query})
        print('result: ', result)

        source_documents = result['source_documents']        
        print('source_documents: ', source_documents)

        if len(source_documents)>=1:
            reference = get_reference(source_documents)
            # print('reference: ', reference)

            return result['result']+reference
        else:
            return result['result']
        
def lambda_handler(event, context):
    print(event)
    userId  = event['user-id']
    print('userId: ', userId)
    requestId  = event['request-id']
    print('requestId: ', requestId)
    type  = event['type']
    print('type: ', type)
    body = event['body']
    print('body: ', body)

    global modelId, llm, kendra
    
    modelId = load_configuration(userId)
    if(modelId==""): 
        modelId = os.environ.get('model_id')
        save_configuration(userId, modelId)

    start = int(time.time())    

    msg = ""
    if type == 'text' and body[:11] == 'list models':
        msg = f"The list of models: \n"
        lists = modelInfo['modelSummaries']
        
        for model in lists:
            msg += f"{model['modelId']}\n"
        
        msg += f"current model: {modelId}"
        print('model lists: ', msg)
    
    elif type == 'text' and body[:20] == 'change the model to ':
        new_model = body.rsplit('to ', 1)[-1]
        print(f"new model: {new_model}, current model: {modelId}")

        if modelId == new_model:
            msg = "No change! The new model is the same as the current model."
        else:        
            lists = modelInfo['modelSummaries']
            isChanged = False
            for model in lists:
                if model['modelId'] == new_model:
                    print(f"new modelId: {new_model}")
                    modelId = new_model
                    llm = Bedrock(model_id=modelId, client=boto3_bedrock)
                    isChanged = True
                    save_configuration(userId, modelId)            

            if isChanged:
                msg = f"The model is changed to {modelId}"
            else:
                msg = f"{modelId} is not in lists."
        print('msg: ', msg)

    else:             
        if type == 'text':
            text = body
            
            querySize = len(text)
            print('query size: ', querySize)

            if querySize<1000: 
                msg = get_answer_using_template(text)
            else:
                msg = llm(text)
            print('msg: ', msg)
            
        elif type == 'document':
            object = body
                    
            # stor the object into kendra
            store_document(object, requestId)

            # summerization to show the content of the document
            file_type = object[object.rfind('.')+1:len(object)]
            print('file_type: ', file_type)
            
            docs = load_document(file_type, object)
            prompt_template = """Write a concise summary of the following:

            {text}
                
            CONCISE SUMMARY """

            PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
            chain = load_summarize_chain(llm, chain_type="stuff", prompt=PROMPT)
            summary = chain.run(docs)
            print('summary: ', summary)

            msg = summary

        elapsed_time = int(time.time()) - start
        print("total run time(sec): ", elapsed_time)

        item = {
            'user-id': {'S':userId},
            'request-id': {'S':requestId},
            'type': {'S':type},
            'body': {'S':body},
            'msg': {'S':msg}
        }

        client = boto3.client('dynamodb')
        try:
            resp =  client.put_item(TableName=callLogTableName, Item=item)
        except: 
            raise Exception ("Not able to write into dynamodb")
        
        print('resp, ', resp)

    return {
        'statusCode': 200,
        'msg': msg,
    }
