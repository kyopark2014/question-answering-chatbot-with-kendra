# CDK를 이용한 인프라 설치하기

여기서는 [Cloud9](https://aws.amazon.com/ko/cloud9/)에서 [AWS CDK](https://aws.amazon.com/ko/cdk/)를 이용하여 인프라를 설치합니다. 현재 Seoul Region에 Kendra 미지원이라서 가장 가까운 Tokyo Region에서 실습을 수행합니다.

Kendra를 설치하기 위해서는 [IAM Identity Center - Tokyo](https://ap-northeast-1.console.aws.amazon.com/singlesignon/home?region=ap-northeast-1#!/instances/users)에서 아래와 같이 최소 하나의 계정이 생성되어 있어야 합니다. 만약 계정이 생성되어 있지 않다면 신규로 생성후 다음 단계를 진행하시기 바랍니다.

![image](https://github.com/kyopark2014/question-answering-chatbot-with-kendra/assets/52392004/8d96153c-98b7-4522-bc31-254d246bc488)


1) [Cloud9 Console](https://ap-northeast-1.console.aws.amazon.com/cloud9control/home?region=ap-northeast-1#/create)에 접속하여 [Create environment]-[Name]에서 “chatbot”으로 이름을 입력하고, EC2 instance는 “m5.large”를 선택합니다. 나머지는 기본값을 유지하고, 하단으로 스크롤하여 [Create]를 선택합니다.

![noname](https://github.com/kyopark2014/chatbot-based-on-Falcon-FM/assets/52392004/7c20d80c-52fc-4d18-b673-bd85e2660850)

2) [Environment](https://ap-northeast-2.console.aws.amazon.com/cloud9control/home?region=ap-northeast-2#/)에서 “chatbot”를 [Open]한 후에 아래와 같이 터미널을 실행합니다.

![noname](https://github.com/kyopark2014/chatbot-based-on-Falcon-FM/assets/52392004/b7d0c3c0-3e94-4126-b28d-d269d2635239)

3) EBS 크기 변경

아래와 같이 스크립트를 다운로드 합니다. 

```text
curl https://raw.githubusercontent.com/kyopark2014/technical-summary/main/resize.sh -o resize.sh
```

이후 아래 명령어로 용량을 100G로 변경합니다.
```text
chmod a+rx resize.sh && ./resize.sh 100
```


4) 소스를 다운로드합니다.

```java
git clone https://github.com/kyopark2014/question-answering-chatbot-with-kendra
```

5) cdk 폴더로 이동하여 필요한 라이브러리를 설치합니다.

```java
cd question-answering-chatbot-with-kendra/cdk-chatbot-with-kendra/ && npm install
```

6) bedrock-sdk를 설치합니다.

```java
cd ../lambda-chat && wget https://preview.documentation.bedrock.aws.dev/Documentation/SDK/bedrock-python-sdk.zip
unzip bedrock-python-sdk.zip -d bedrock-sdk && python3 -m pip install -U ./bedrock-sdk/boto3-1.26.162-py3-none-any.whl -t . && python3 -m pip install -U ./bedrock-sdk/awscli-1.27.162-py3-none-any.whl -t . && cd ../cdk-chatbot-with-kendra/
```

6) CDK 사용을 위해 Boostraping을 수행합니다.

아래 명령어로 Account ID를 확인합니다.

```java
aws sts get-caller-identity --query Account --output text
```

아래와 같이 bootstrap을 수행합니다. 여기서 "account-id"는 상기 명령어로 확인한 12자리의 Account ID입니다. bootstrap 1회만 수행하면 되므로, 기존에 cdk를 사용하고 있었다면 bootstrap은 건너뛰어도 됩니다.

```java
cdk bootstrap aws://account-id/ap-northeast-1
```

8) 인프라를 설치합니다.

```java
cdk deploy
```
9) 설치가 완료되면 브라우저에서 아래와 같이 WebUrl를 확인하여 브라우저를 이용하여 접속합니다.

![noname](https://github.com/kyopark2014/chatbot-based-on-bedrock-anthropic/assets/52392004/369b175f-9bd5-4e34-ad0e-e9f2a50e90fb)
