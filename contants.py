from langchain.utils import get_from_env

PINECONE_KEY='07134ff5-efff-48bc-9910-37cdf1eaf687'
PINECONE_ENV='us-central1-gcp'
QDRANT_API_KEY = 'chODw5UIjCvNhdXSXWu7DCGusyEjlw4lzX3XwD6aPHsLGYvumAutWg'
QDRANT_URL = 'https://c5eaad32-9eab-4557-8115-500b6031ade8.us-east-1-0.aws.cloud.qdrant.io:6333'

QA_SYSTEM_PROMPT_BLOCK_UNRELATED = """You are a helpful and empathetic coach that gives actionable advice and helps users achieve goals.
You are designed to answer questions only from known sets of documents, DO NOT USE general knowledge.
You have access to tools for interacting with the documents, and the inputs to the tools are questions.
You will provide sources for your questions without asking, in which case you should use the appropriate tool to do so.
If the question does not seem relevant to any of the tools provided, just return "I don't know" as the answer.
Analyze the user's input and the conversation. Do not repeat on what the conversation has already covered.
Please ask one question to follow up for better understand the situation if needed or ask one powerful question to help users generate insights, or give actionable advice based on the User's input.

If the input is talking about user's feeling or emotion, please ask the user to elaborate more like "What makes you feel that way?" or "What's the reason behind that?".
You must provide a follow up question at last for the user to ask next to continue the conversation.
Prioritize the response based on the user's input, then the conversation and the user's information (goals, challenges).
Don't use the dialog format and answer in plain text only. Response should be short and concise with COMPLETE SENTENCES.

Bulletin format is always a preferred format."""

QA_SYSTEM_PROMPT = """You are a helpful and empathetic coach that gives actionable advice and helps users achieve goals. You have access to tools for interacting with the documents. You will provide sources for your responses without asking, in which case you should use the appropriate tool to do so.
Do not repeat on what the conversation has already covered.
Please ask one question to follow up for better understand the situation if needed or ask one powerful question to help users generate insights, or give actionable advice based on the User's input.

If the input is talking about user's feeling or emotion, please ask the user to elaborate more like "What makes you feel that way?" or "What's the reason behind that?".
You must provide a follow up question at last for the user to ask next to continue the conversation.
Prioritize the response based on the user's input, then the conversation and the user's information (goals, challenges).
Don't use the dialog format and answer in plain text only. Response should be short and concise with COMPLETE SENTENCES.
"""

G_CERT = {
    "type": "service_account",
    "project_id": "amotions-web",
    "private_key_id": "64d68dad94ad0fe6594d9e42e6137bb4b63406f1",
    "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQDbiKdO/2zyntcP\nQbCU5dOpQn94Lokh/+cV3VbSQgOKTUHs3Tzxcs2OaBRGsVebsfWsq1gqIFj/B+5N\nf/HK9GwV99KbFSOSB9yylr3+/+rMncorZBj7Cvt2GxvniPXcGRQXBrTeJ1X/GZB4\nbpch85t9A+2dO20wiDCVsicaOMPn9o+E9cNiHOAd2NOLhDs2u4noTc/PsMT0TwyI\n70t4vmNmumdD6HE08Wd42FKbffCblu8t7JhsLW6eq/mWbAE3URmTiXPE0AS7sqiE\nn0rFBZzZFpJAVdVtvywRrQVzRsec0RpjhiEnWOs+2mM11SznEPEB4lrNgtPEtrwz\nJxOAEheBAgMBAAECggEABvqvsvu813PES4hfmEQ8N+fUzJnSI0IS8yd/wJx3p1hw\nbZrU+qoXLd1SM7MkoV8Fe/FQkrpHiggTA+S656RgoUwpJmQEJvrYAdd3/9jb+phT\nmpCKmMSPHQCcoP5CLrafZGICswNWIQ+Lf8CoTlyDx3QuBC8k+AOEY+Xvh05hsAig\nB4G6b1eErn+Za+vy9/IGJamloJo+kc9DCI1CzNMlZpYvqB6zTlbuG/h/6wvdmPhN\nh0nY/4fNxNpHBPoePxGYzLmgbBQDmbZpW6J/BkYUzYA71yDhe6IKlafHXZp6W0YB\nzCl5TaCBXbfAYBE1o6A0ac1tTGlq8XtYnjhKVKhR5wKBgQD+256nfHUl7nIAVkf7\nMVf5RKdjpIB3mLk0jqz8TobbRm7Thx/vFmbJFnP++rpJnKRn1DnR4VcOOwP2vL9+\nBomu/FdtoVNnWpuzyjtTuydkvDovKbfflEDs8sBAfa0FOtPjPsmS1k6YRxz6d1yM\n3dkCSWi5OroXz8EyJgkSvki0OwKBgQDchIJOFLhfkPzMQ2GduPbvt8PAu/IMG2gI\nSde80OWcbry1606hUssxeN0fKCHgRgo+K4OuWhUQoJgde1qnSpnZAuZw04GRrcJZ\n5+6LAKPI93FDhxPXMRi6pIBlMlFVDxGKrKi3dzcMuFMNuDHD33Y5DmGJQ9A/QYh6\nzqYjC05TcwKBgQDB/q6v0t7hdrW0Z/j2zkKm7Yl1IZzgbJJd3VTz4Vppwx1NSogW\nWDj5TGwTZQs0SVYj6rnwdtTrciS8RkSFS7i2SELMooZ4H9JxcrjiLY348gZgLCNY\nvuME+ms5K/DuEC+FxR9u9E5zP35hcUYzvBMZ2IMsq/VHds8auzUg3VM+TQKBgCAE\n7ZbA03Ss7MgMEKSCMvjjyfy2TZMMd7KcZkL4Yh6wxZ30qNor1207i3/2p9SU1u/o\nCZrLYbukVfIR8zvPTT+BeExxqaKphErhrnVohl+r7jpL5smcS2buc+Y9xCmBop5s\nK6NYIBosuYKIeFFkpRnryKXRvu8waMzMLLtx7NGZAoGAXxlkQ0CcOWb4KmOXbM+/\nL2TgIQOW6ciD6wTLdipNxSqly0ukXscPLRsFIXex9RGQf1hZ864sqzV1T1PkC5YT\n8xDz68kffNMh/A+wCij1bf/4RuID/+4U6qtMRe1Ewp54HKoP11MukomQ6Sz/P+5o\nSpAZcgyiIQveNlforSqkq4k=\n-----END PRIVATE KEY-----\n",
    "client_email": "firebase-adminsdk-k40gb@amotions-web.iam.gserviceaccount.com",
    "client_id": "105415921885956808595",
    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
    "token_uri": "https://oauth2.googleapis.com/token",
    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
    "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/firebase-adminsdk-k40gb%40amotions-web.iam.gserviceaccount.com",
    "universe_domain": "googleapis.com"
}
