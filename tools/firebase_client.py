
import re
from typing import List, Optional, Type
from firebase_admin import firestore
from google.cloud.firestore_v1.base_query import FieldFilter
from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.tools import BaseTool
from langchain.pydantic_v1 import BaseModel, Field
from langchain import OpenAI


class FirebaseClient:
    def __init__(self) -> None:
        self.db = firestore.client()
        self._videos = self._all_data('videos')
        self._skills = self._all_data('skills')

    def _all_data(self, collection):
        data = []
        for doc in self.db.collection(collection).stream():
            item = doc.to_dict()
            item.update(id=doc.id)
            data.append(item)
        return data

    @property
    def all_videos(self):
        return self._videos

    @property
    def all_skills(self):
        return self._skills

    def user_chat_history(self, uid):
        pass


# class FirebaseVideosTool(BaseTool):
#     name: str = "amotions_videos"
#     description: str = """This tool can do 2 things:
#     1. return all available videos.
#     2. load video by title"""
#     client: FirebaseClient = Field(default_factory=FirebaseClient)

#     def _run(
#         self,
#         query: str,
#         run_manager: Optional[CallbackManagerForToolRun] = None,
#     ) -> str:
#         return "|".join(list(map(lambda x: x['title'], self.client.all_videos)))


# class FirebaseSkillsTool(BaseTool):
#     name: str = "amotions_skills"
#     description: str = """This tool can do 2 things:
#     1. return all available skills and practices.
#     2. load  skills and practices by name"""
#     client: FirebaseClient = Field(default_factory=FirebaseClient)

#     def _run(
#         self,
#         query: str,
#         run_manager: Optional[CallbackManagerForToolRun] = None,
#     ) -> str:
#         return '|'.join(list(map(lambda x: x['name'], self.client.all_skills)))


class RecommendVideoTool(BaseTool):
    name: str = "amotions_recommend_videos"
    description: str = """search videos for user"""
    llm = OpenAI(temperature=0)
    client: FirebaseClient = Field(default_factory=FirebaseClient)

    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        videos = '\n'.join(map(lambda x: x['title'], self.client.all_videos))
        prompt = f"""Analyze the below videos to provide suggestions".

                The videos are as follows:

                {videos}

                Based on the user's input and the videos suggest 1 video only. If none of the videos are related to the User Input, say you don't have relevant videos about the topic. Don't make up videos - Only use the videos in the list above.
                You only give video back without other content
                
                user's input is [{query}]."""
        ret = self.llm(prompt)

        title = ret.strip()
        output = ""
        for v in self.client.all_videos:
            if v['title'].lower() in title.lower():
                output = f"""[{title}](https://www.amotionsinc.com/video/{v['id']})"""
        return output


if __name__ == '__main__':
    firebase_cli = FirebaseClient(
        "amotions-web-firebase-adminsdk-k40gb-64d68dad94.json")
    for v in firebase_cli.all_videos:
        print(v['title'])
        print(v['transcript'])
        print('\n')
