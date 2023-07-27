# Amazon Kendra를 이용한 Question/Answering Chatbot 만들기 

여기서는 Amazon Bedrock의 LLM 모델을 이용하여 Question/Answering을 수행하는 Chatbot을 만듧니다. Question/Answering의 정확도를 높이기 위하여 문서를 업로드하여 Kendra로 분석하고, LLM이 Question/Answering 수행시 kendra에 있는 index db를 통해 가장 유사한 문서를 추출하여 사용할 수 있습니다. 이렇게 vector store를 사용하면 LLM의 token 사이즈를 넘어서는 긴문장을 활용하여 Question/Answering과 같은 Task를 수행할 수 있으며 환각(hallucination) 영향을 줄일 수 있습니다.

전체적인 Architecture는 아래와 같습니다. 사용자가 파일을 로드하면 CloudFont와 API Gateway를 거쳐서 [Lambda (upload)](./lambda-upload/index.js)가 S3에 파일을 저장합니다. 저장이 완료되면 해당 Object의 bucket과 key를 이용하여 [Lambda (chat)](./lambda-chat/lambda_function.py)이 파일을 로드하여 text를 추출합니다. text는 chunk size로 분리되어서 embedding을 통해 vector store에 index로 저장됩니다. 사용자가 메시지를 전달하면 vector store로 부터 가장 가까운 chunk들을 이용하여 Question/Answering을 수행합니다. 이후 관련된 call log는 DynamoDB에 저장됩니다. 여기서 LLM은 Bedrock을 LangChain 형식의 API를 통해 구현하였고, Chatbot을 제공하는 인프라는 AWS CDK를 통해 배포합니다. 

<img src="https://github.com/kyopark2014/question-answering-chatbot-with-kendra/assets/52392004/d4ce692f-6a6b-45ca-8ebe-7427b1e9b601" width="750">

## 주요 구성

### IAM Role

IAM Role에 아래와 같은 Permission을 추가해야 합니다.

```java
{
    "Effect": "Allow",
    "Action": [
        "kendra:*"
    ],
    "Resource": "arn:aws:kendra:[your-region]:[your-account-id]:index/[index-id]"
}]
```

이를 [cdk-chatbot-with-kendra-stack.ts](./cdk-chatbot-with-kendra/lib/cdk-chatbot-with-kendra-stack.ts)에서는 아래와 구현할 수 있습니다.

```java
const region = process.env.CDK_DEFAULT_REGION;
const accountId = process.env.CDK_DEFAULT_ACCOUNT;
const kendraResourceArn = `arn:aws:kendra:${region}:${accountId}:index/${kendraIndex}`
if (debug) {
    new cdk.CfnOutput(this, `resource-arn-of-kendra-for-${projectName}`, {
        value: kendraResourceArn,
        description: 'The arn of resource',
    });
}
const kendraPolicy = new iam.PolicyStatement({
    resources: [kendraResourceArn],
    actions: ['kendra:*'],
});

roleLambda.attachInlinePolicy( // add kendra policy
    new iam.Policy(this, `kendra-policy-for-${projectName}`, {
        statements: [kendraPolicy],
    }),
);  
```

Kendra를 위한 trust policy는 아래와 같이 설정합니다.

```java
{
   "Version":"2012-10-17",
   "Statement":[
      {
         "Effect":"Allow",
         "Principal":{
            "Service":"kendra.amazonaws.com"
         },
         "Action":"sts:AssumeRole"
      }
   ]
}
```

따라서, [cdk-chatbot-with-kendra-stack.ts](./cdk-chatbot-with-kendra/lib/cdk-chatbot-with-kendra-stack.ts)와 같이 "kendra.amazonaws.com"을 추가합니다.

```java
const roleLambda = new iam.Role(this, `role-lambda-chat-for-${projectName}`, {
    roleName: `role-lambda-chat-for-${projectName}`,
    assumedBy: new iam.CompositePrincipal(
        new iam.ServicePrincipal("lambda.amazonaws.com"),
        new iam.ServicePrincipal("bedrock.amazonaws.com"),
        new iam.ServicePrincipal("kendra.amazonaws.com")
    )
});
```

[Troubleshooting Amazon Kendra Identity and Access](https://docs.aws.amazon.com/kendra/latest/dg/security_iam_troubleshoot.html)와 같아 Kendra는 "iam:PassRole"을 포함하여야 합니다. 

```java
{
    "Action": [
        "iam:PassRole"
    ],
    "Resource": [
        "arn:aws:iam::[account-id]:role/role-lambda-chat-for-chatbot-with-kendra",
    ],
    "Effect": "Allow"
}
```

이를 [cdk-chatbot-with-kendra-stack.ts](./cdk-chatbot-with-kendra/lib/cdk-chatbot-with-kendra-stack.ts)에서는 아래와 같이 구현할 수 있습니다.

```java
const passRoleResourceArn = roleLambda.roleArn;
const passRolePolicy = new iam.PolicyStatement({
    resources: [passRoleResourceArn],
    actions: ['iam:PassRole'],
});
roleLambda.attachInlinePolicy(
    new iam.Policy(this, `pass-role-of-kendra-for-${projectName}`, {
        statements: [passRolePolicy],
    }),
);
```  

### Bedrock을 LangChain으로 연결하기

Bedrock 접속을 위해 필요한 region name과 endpoint url을 지정하고, LangChain을 사용할 수 있도록 연결하여 줍니다. Bedrock preview에서는 Dev/Prod 버전에 따라 endpoint를 달리하는데, Prod 버전을 사용하고자 할 경우에는 endpoint에 대한 부분을 삭제하거나 주석처리합니다.

```python
from langchain.llms.bedrock import Bedrock

bedrock_region = "us-west-2" 
bedrock_config = {
    "region_name":bedrock_region,
    "endpoint_url":"https://prod.us-west-2.frontend.bedrock.aws.dev"
}
    
boto3_bedrock = bedrock.get_bedrock_client(
    region=bedrock_config["region_name"],
    url_override=bedrock_config["endpoint_url"])

modelId = 'amazon.titan-tg1-large'  # anthropic.claude-v1
llm = Bedrock(model_id=modelId, client=boto3_bedrock)    
```


### Embedding

BedrockEmbeddings을 이용하여 Embedding을 합니다.

```python
from langchain.embeddings import BedrockEmbeddings
bedrock_embeddings = BedrockEmbeddings(client=boto3_bedrock)
```

### Kendra

### 문서 등록

S3에 저장된 문서를 kendra로 전달하기 위하여, 아래와 같이 문서에 대한 S3 정보를 kendra의 [batch_put_document()](https://docs.aws.amazon.com/kendra/latest/APIReference/API_BatchPutDocument.html)을 이용하여 전달합니다. 

```python
documentInfo = {
    "S3Path": {
        "Bucket": s3_bucket,
        "Key": s3_prefix+'/'+s3_file_name
    },
    "Title": "Document from client",
    "Id": requestId
}

documents = [
    documentInfo
]

kendra = boto3.client("kendra")
kendra.batch_put_document(
    Documents = documents,
    IndexId = kendraIndex,
    RoleArn = roleArn
)
```

업로드한 문서 파일에 대한 정보를 사용자에게 보여주기 위하여 아래와 같이 요약(Summerization)을 수행합니다.

```python
file_type = object[object.rfind('.') + 1: len(object)]
print('file_type: ', file_type)

docs = load_document(file_type, object)
prompt_template = """Write a concise summary of the following:

{ text }
                
CONCISE SUMMARY """

PROMPT = PromptTemplate(template = prompt_template, input_variables = ["text"])
chain = load_summarize_chain(llm, chain_type = "stuff", prompt = PROMPT)
summary = chain.run(docs)
print('summary: ', summary)

msg = summary
```


### Question/Answering

사용자가 채팅창에서 메시지를 입력할때 발생한 메시지는 아래처럼 query로 전달되고, [Kendra Retriever](https://python.langchain.com/docs/integrations/retrievers/amazon_kendra_retriever)를 이용하여 get_relevant_documents()로 관련된 문장들을 kendra로부터 가져옵니다. 이때 가져온 문장이 없다면 bedrock의 llm()을 이용하여 결과를 얻고, kendra에 관련된 데이터가 있다면 아래와 같이 template을 이용하여 [RetrievalQA](https://python.langchain.com/docs/modules/chains/popular/vector_db_qa)로 query에 대한 응답을 구하여 결과로 전달합니다.

```python
relevant_documents = retriever.get_relevant_documents(query)

if (len(relevant_documents) == 0):
    return llm(query)
else:
    prompt_template = """Human: Use the following pieces of context to provide a concise answer to the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

    { context }

    Question: { question }
    Assistant: """
    PROMPT = PromptTemplate(
        template = prompt_template, input_variables = ["context", "question"]
    )

    qa = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type = "stuff",
        retriever = retriever,
        return_source_documents = True,
        chain_type_kwargs = { "prompt": PROMPT }
    )
    result = qa({ "query": query })

    return result['result']
```

## 실습하기

### CDK를 이용한 인프라 설치
[인프라 설치](https://github.com/kyopark2014/question-answering-chatbot-using-RAG-based-on-LLM/blob/main/deployment.md)에 따라 CDK로 인프라 설치를 진행합니다. [CDK 구현 코드](./cdk-qa-with-rag/README.md)에서는 Typescript로 인프라를 정의하는 방법에 대해 상세히 설명하고 있습니다.


## 실행결과

파일을 올리면 파일의 텍스트를 기반으로 요약(Summeraztion)을 수행합니다.

![image](https://github.com/kyopark2014/question-answering-chatbot-using-RAG-based-on-LLM/assets/52392004/93a8d391-905b-487b-a45e-a7c67f03528b)

이후 텍스트로 질문을 하면 아래와 같이 업로드한 문서파일을 기반으로 답변을 수행합니다.

![image](https://github.com/kyopark2014/question-answering-chatbot-using-RAG-based-on-LLM/assets/52392004/4050847f-8f05-4136-a290-9818f108d1cb)

잘못된 질문을 하여도 아래와 같이 업로드한 문서에 맞추어서 답변을 할 수 있습니다.

![image](https://github.com/kyopark2014/question-answering-chatbot-using-RAG-based-on-LLM/assets/52392004/4be7f868-b830-4e2f-af1d-445f905f280b)

## 브라우저에서 Chatbot 동작 시험시 주의할점

Chatbot API를 테스트하기 위해 제공하는 Web client는 일반적인 채팅 App처럼 세션 방식(websocket등)이 아니라 RESTful API를 사용합니다. 따라서 아래와 같은 특징이 있습니다.

1) LLM에서 응답이 일정시간(30초)이상 지연되는 경우에 답변을 볼 수 없습니다. 브라우저 자체적인 timeout으로 인하여 30초 이상인 경우에 client에서 더이상 응답을 받을 수 없습니다. 이때 응답을 확인하기 위해서는 CloudWatch에서 [lambda-chat](./lambda-chat/lambda_function.py)의 로그를 확인하거나, DynamoDB에 저장된 call log를 확인합니다.
2) 한번에 전달할 수 있는 Lambda Payload 사이즈 제한으로 5MB이하의 파일만 업로드가 가능합니다. 


## Reference 

[Kendra - LangChain](https://python.langchain.com/docs/integrations/retrievers/amazon_kendra_retriever)

[kendra_chat_anthropic.py](https://github.com/aws-samples/amazon-kendra-langchain-extensions/blob/main/kendra_retriever_samples/kendra_chat_anthropic.py)

[IAM access roles for Amazon Kendra](https://docs.aws.amazon.com/kendra/latest/dg/iam-roles.html)

[Adding documents with the BatchPutDocument API](https://docs.aws.amazon.com/kendra/latest/dg/in-adding-binary-doc.html)
