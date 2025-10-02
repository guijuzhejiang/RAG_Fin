#
#  Copyright 2024 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
from openai.lib.azure import AzureOpenAI
from zhipuai import ZhipuAI
import io
from abc import ABC
from ollama import Client
from PIL import Image
from openai import OpenAI
import os
import base64
from io import BytesIO
import json
import requests
import gc
from rag.nlp import is_english
from api.utils import get_uuid
from api.utils.file_utils import get_project_base_directory
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava15ChatHandler


class Base(ABC):
    def __init__(self, key, model_name):
        pass

    def describe(self, image, max_tokens=300):
        raise NotImplementedError("Please implement encode method!")
        
    def chat(self, system, history, gen_conf, image=""):
        if system:
            history[-1]["content"] = system + history[-1]["content"] + "user query: " + history[-1]["content"]
        try:
            for his in history:
                if his["role"] == "user":
                    his["content"] = self.chat_prompt(his["content"], image)

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=history,
                max_tokens=gen_conf.get("max_tokens", 1000),
                temperature=gen_conf.get("temperature", 0.3),
                top_p=gen_conf.get("top_p", 0.7)
            )
            return response.choices[0].message.content.strip(), response.usage.total_tokens
        except Exception as e:
            return "**ERROR**: " + str(e), 0

    def chat_streamly(self, system, history, gen_conf, image=""):
        if system:
            history[-1]["content"] = system + history[-1]["content"] + "user query: " + history[-1]["content"]

        ans = ""
        tk_count = 0
        try:
            for his in history:
                if his["role"] == "user":
                    his["content"] = self.chat_prompt(his["content"], image)

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=history,
                max_tokens=gen_conf.get("max_tokens", 1000),
                temperature=gen_conf.get("temperature", 0.3),
                top_p=gen_conf.get("top_p", 0.7),
                stream=True
            )
            for resp in response:
                if not resp.choices[0].delta.content: continue
                delta = resp.choices[0].delta.content
                ans += delta
                if resp.choices[0].finish_reason == "length":
                    ans += "...\nFor the content length reason, it stopped, continue?" if is_english(
                        [ans]) else "······\n回答が長すぎるため、一部が省略されました。続きを表示しますか？"
                    tk_count = resp.usage.total_tokens
                if resp.choices[0].finish_reason == "stop": tk_count = resp.usage.total_tokens
                yield ans
        except Exception as e:
            yield ans + "\n**ERROR**: " + str(e)

        yield tk_count
        
    def image2base64(self, image):
        if isinstance(image, bytes):
            return base64.b64encode(image).decode("utf-8")
        if isinstance(image, BytesIO):
            return base64.b64encode(image.getvalue()).decode("utf-8")
        
        buffered = BytesIO()
        # 直接用PNG（支持所有格式包括RGBA）
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def prompt(self, b64):
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{b64}"
                        },
                    },
                    {
                        "text": '''You are analyzing an image that may contain charts, graphs, infographics, tables, maps, or other visual representations. Your task is to provide an accurate, concise, and RAG-ready description focused on factual content suitable for a knowledge base.

SPECIFIC INSTRUCTIONS:
1. CAREFULLY EXAMINE THE IMAGE CONTENT:
   - First determine the image type (financial chart, business graph, map, diagram, table, or other)
   - Adapt your analysis approach based on the image type

2. FOR FINANCIAL/BUSINESS IMAGES (charts, graphs, tables):
   - Focus on numerical data: exact numbers, percentages, monetary values, dates and units.
   - Identify and report key financial/business metrics (revenue, profit, growth rates, market share, KPIs) only if clearly visible.
   - Highlight trends, comparisons, and significant changes using concise bullets.
   - Always report time periods, measurement units, and currency types when present.

3. FOR NON-BUSINESS IMAGES (maps, diagrams, general visuals):
   - Provide a factual, concise description of the depicted elements and relationships.
   - Extract any visible text, labels, or numbers.
   - Report spatial or causal relationships if clearly shown.
   - Do not force a financial/business interpretation.

4. DATA EXTRACTION PRINCIPLES:
   - Do not invent, assume, or extrapolate information that is not visibly present.
   - Report only clearly visible and readable information
   - Do not include decorative/visual attributes (color, shape, font, layout) unless they convey real data (e.g., legend mapping color → value). Descriptions like “blue oval” or “green box” are unnecessary for the knowledge base and must be omitted.

5. OUTPUT FORMAT (RAG-FRIENDLY):
   - Produce a short, factual summary suitable for ingestion into a knowledge base.
   - Each bullet must be a single factual statement (no narrative fluff)
   - For data-rich images: use bullet points for key findings
   - Always maintain factual accuracy over speculative interpretation

6. QUALITY CONTROL:
   - If the image is decorative or purely artistic without data, give one brief factual sentence (no numeric analysis).

PRIMARY GOAL: produce a compact, factual, non-decorative description optimized for RAG ingestion — prioritize numbers and explicit claims; otherwise summarize clearly stated relationships without inventing details.
''',
                    },
                ],
            }
        ]

    def chat_prompt(self, text, b64):
        return [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{b64}",
                },
            },
            {
                "type": "text",
                "text": text
            },
        ]


class GptV4(Base):
    def __init__(self, key, model_name="gpt-4-vision-preview", lang="Chinese", base_url="https://api.openai.com/v1"):
        if not base_url: base_url="https://api.openai.com/v1"
        self.client = OpenAI(api_key=key, base_url=base_url)
        self.model_name = model_name
        self.lang = lang

    def describe(self, image, max_tokens=300):
        b64 = self.image2base64(image)
        prompt = self.prompt(b64)
        for i in range(len(prompt)):
            for c in prompt[i]["content"]:
                if "text" in c: c["type"] = "text"

        res = self.client.chat.completions.create(
            model=self.model_name,
            messages=prompt,
            max_tokens=max_tokens,
        )
        return res.choices[0].message.content.strip(), res.usage.total_tokens

class AzureGptV4(Base):
    def __init__(self, key, model_name, lang="Chinese", **kwargs):
        self.client = AzureOpenAI(api_key=key, azure_endpoint=kwargs["base_url"], api_version="2024-02-01")
        self.model_name = model_name
        self.lang = lang

    def describe(self, image, max_tokens=300):
        b64 = self.image2base64(image)
        prompt = self.prompt(b64)
        for i in range(len(prompt)):
            for c in prompt[i]["content"]:
                if "text" in c: c["type"] = "text"

        res = self.client.chat.completions.create(
            model=self.model_name,
            messages=prompt,
            max_tokens=max_tokens,
        )
        return res.choices[0].message.content.strip(), res.usage.total_tokens


class QWenCV(Base):
    def __init__(self, key, model_name="qwen-vl-chat-v1", lang="Chinese", **kwargs):
        import dashscope
        dashscope.api_key = key
        self.model_name = model_name
        self.lang = lang

    def prompt(self, binary):
        # stupid as hell
        tmp_dir = get_project_base_directory("tmp")
        if not os.path.exists(tmp_dir):
            os.mkdir(tmp_dir)
        path = os.path.join(tmp_dir, "%s.jpg" % get_uuid())
        Image.open(io.BytesIO(binary)).save(path)
        return [
            {
                "role": "user",
                "content": [
                    {
                        "image": f"file://{path}"
                    },
                    {
                        "text": "请用中文详细描述一下图中的内容，比如时间，地点，人物，事情，人物心情等，如果有数据请提取出数据。" if self.lang.lower() == "chinese" else
                        "Please describe the content of this picture, like where, when, who, what happen. If it has number data, please extract them out.",
                    },
                ],
            }
        ]

    def chat_prompt(self, text, b64):
        return [
            {"image": f"{b64}"},
            {"text": text},
        ]
    
    def describe(self, image, max_tokens=300):
        from http import HTTPStatus
        from dashscope import MultiModalConversation
        response = MultiModalConversation.call(model=self.model_name,
                                               messages=self.prompt(image))
        if response.status_code == HTTPStatus.OK:
            return response.output.choices[0]['message']['content'][0]["text"], response.usage.output_tokens
        return response.message, 0

    def chat(self, system, history, gen_conf, image=""):
        from http import HTTPStatus
        from dashscope import MultiModalConversation
        if system:
            history[-1]["content"] = system + history[-1]["content"] + "user query: " + history[-1]["content"]

        for his in history:
            if his["role"] == "user":
                his["content"] = self.chat_prompt(his["content"], image)
        response = MultiModalConversation.call(model=self.model_name, messages=history,
                                               max_tokens=gen_conf.get("max_tokens", 1000),
                                               temperature=gen_conf.get("temperature", 0.3),
                                               top_p=gen_conf.get("top_p", 0.7))

        ans = ""
        tk_count = 0
        if response.status_code == HTTPStatus.OK:
            ans += response.output.choices[0]['message']['content']
            tk_count += response.usage.total_tokens
            if response.output.choices[0].get("finish_reason", "") == "length":
                ans += "...\nFor the content length reason, it stopped, continue?" if is_english(
                    [ans]) else "······\n由于长度的原因，回答被截断了，要继续吗？"
            return ans, tk_count

        return "**ERROR**: " + response.message, tk_count

    def chat_streamly(self, system, history, gen_conf, image=""):
        from http import HTTPStatus
        from dashscope import MultiModalConversation
        if system:
            history[-1]["content"] = system + history[-1]["content"] + "user query: " + history[-1]["content"]

        for his in history:
            if his["role"] == "user":
                his["content"] = self.chat_prompt(his["content"], image)

        ans = ""
        tk_count = 0
        try:
            response = MultiModalConversation.call(model=self.model_name, messages=history,
                                                   max_tokens=gen_conf.get("max_tokens", 1000),
                                                   temperature=gen_conf.get("temperature", 0.3),
                                                   top_p=gen_conf.get("top_p", 0.7),
                                                   stream=True)
            for resp in response:
                if resp.status_code == HTTPStatus.OK:
                    ans = resp.output.choices[0]['message']['content']
                    tk_count = resp.usage.total_tokens
                    if resp.output.choices[0].get("finish_reason", "") == "length":
                        ans += "...\nFor the content length reason, it stopped, continue?" if is_english(
                            [ans]) else "······\n由于长度的原因，回答被截断了，要继续吗？"
                    yield ans
                else:
                    yield ans + "\n**ERROR**: " + resp.message if str(resp.message).find(
                        "Access") < 0 else "Out of credit. Please set the API key in **settings > Model providers.**"
        except Exception as e:
            yield ans + "\n**ERROR**: " + str(e)

        yield tk_count


class Zhipu4V(Base):
    def __init__(self, key, model_name="glm-4v", lang="Chinese", **kwargs):
        self.client = ZhipuAI(api_key=key)
        self.model_name = model_name
        self.lang = lang

    def describe(self, image, max_tokens=1024):
        b64 = self.image2base64(image)

        prompt = self.prompt(b64)
        prompt[0]["content"][1]["type"] = "text"
        
        res = self.client.chat.completions.create(
            model=self.model_name,
            messages=prompt,
            max_tokens=max_tokens,
        )
        return res.choices[0].message.content.strip(), res.usage.total_tokens

    def chat(self, system, history, gen_conf, image=""):
        if system:
            history[-1]["content"] = system + history[-1]["content"] + "user query: " + history[-1]["content"]
        try:
            for his in history:
                if his["role"] == "user":
                    his["content"] = self.chat_prompt(his["content"], image)

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=history,
                max_tokens=gen_conf.get("max_tokens", 1000),
                temperature=gen_conf.get("temperature", 0.3),
                top_p=gen_conf.get("top_p", 0.7)
            )
            return response.choices[0].message.content.strip(), response.usage.total_tokens
        except Exception as e:
            return "**ERROR**: " + str(e), 0

    def chat_streamly(self, system, history, gen_conf, image=""):
        if system:
            history[-1]["content"] = system + history[-1]["content"] + "user query: " + history[-1]["content"]

        ans = ""
        tk_count = 0
        try:
            for his in history:
                if his["role"] == "user":
                    his["content"] = self.chat_prompt(his["content"], image)

            response = self.client.chat.completions.create(
                model=self.model_name, 
                messages=history,
                max_tokens=gen_conf.get("max_tokens", 1000),
                temperature=gen_conf.get("temperature", 0.3),
                top_p=gen_conf.get("top_p", 0.7),
                stream=True
            )
            for resp in response:
                if not resp.choices[0].delta.content: continue
                delta = resp.choices[0].delta.content
                ans += delta
                if resp.choices[0].finish_reason == "length":
                    ans += "...\nFor the content length reason, it stopped, continue?" if is_english(
                        [ans]) else "······\n由于长度的原因，回答被截断了，要继续吗？"
                    tk_count = resp.usage.total_tokens
                if resp.choices[0].finish_reason == "stop": tk_count = resp.usage.total_tokens
                yield ans
        except Exception as e:
            yield ans + "\n**ERROR**: " + str(e)

        yield tk_count


class OllamaCV(Base):
    def __init__(self, key, model_name, lang="English", **kwargs):
        self.client = Client(host=kwargs["base_url"])
        self.model_name = model_name
        self.lang = lang

    def describe(self,
                 image,
                 context: str = '',
                 num_predict: int = 512,
                 temperature: float = 0.6,
                 top_p: float = 0.9,
                 top_k: int = 40,
                 repeat_penalty: float = 1.1):
        # b64 = self.image2base64(image)
        prompt_template = '''You are analyzing an image that may contain charts, graphs, infographics, tables, maps, or other visual representations. Your task is to provide an accurate, concise, and RAG-ready description focused on factual content suitable for a knowledge base.

CONTEXT INTEGRATION:
1. You have been provided with context from the PDF document: {context}
2. FIRST, evaluate whether this context is relevant to the image content. 
3. If the context is RELEVANT, integrate meaningful information from it to enhance your analysis
4. If the context is IRRELEVANT, prioritize the actual image content and ignore the context

SPECIFIC INSTRUCTIONS:
1. CAREFULLY EXAMINE THE IMAGE CONTENT:
   - First determine the image type (financial chart, business graph, map, diagram, table, or other)
   - Adapt your analysis approach based on the image type

2. FOR FINANCIAL/BUSINESS IMAGES (charts, graphs, tables):
   - First identify x-axis and y-axis meaning (e.g., x = year, y = monetary value). If y-axis ticks and unit are visible (e.g., "HK$M" and tick labels), include the unit in all reported values.
   - If the chart is stacked, attempt to extract values per legend item. If numeric labels are not present, perform linear interpolation between visible y-axis ticks.
   - Focus on numerical data: exact numbers, percentages, monetary values, dates and units.
   - Identify and report key financial/business metrics (revenue, profit, growth rates, market share, KPIs) only if clearly visible.
   - Highlight trends, comparisons, and significant changes using concise bullets.
   - Always report time periods, measurement units, and currency types when present.
   - For charts that use a legend (e.g., stacked bars or colored segments), map legend entries to visual segments by matching legend labels to visual indicators (prefer pixel/colour sampling, e.g., RGB/HEX) rather than vague color names. Use that mapping to assign each visible segment to a label/region.
   - If numeric labels are present on bars/segments/points, treat those as "measured". If numeric labels are absent but axis ticks and units are readable, convert pixel heights/positions to data values via linear interpolation between visible ticks.
   - Output format guidance: produce a machine-readable structured output (JSON or equivalent) with one record per x-tick. 

3. FOR NON-BUSINESS IMAGES (maps, diagrams, general visuals):
   - Provide a factual, concise description of the depicted elements and relationships.
   - Extract any visible text, labels, or numbers.
   - Report spatial or causal relationships if clearly shown.
   - Do not force a financial/business interpretation.

4. DATA EXTRACTION PRINCIPLES:
   - Do not invent, assume, or extrapolate information that is not visibly present.
   - Report only clearly visible and readable information
   - Do not include decorative/visual attributes (color, shape, font, layout) unless they convey real data (e.g., legend mapping color → value). Descriptions like “blue oval” or “green box” are unnecessary for the knowledge base and must be omitted.

5. OUTPUT FORMAT (RAG-FRIENDLY):
   - Produce a short, factual summary suitable for ingestion into a knowledge base.
   - Each bullet must be a single factual statement (no narrative fluff)
   - For data-rich images: use bullet points for key findings
   - Always maintain factual accuracy over speculative interpretation

6. QUALITY CONTROL:
   - If the image is decorative or purely artistic without data, give one brief factual sentence (no numeric analysis).

PRIMARY GOAL: produce a compact, factual, non-decorative description optimized for RAG ingestion — prioritize numbers and explicit claims; otherwise summarize clearly stated relationships without inventing details.
'''
        
        # 使用 format 方法将 context 参数插入 prompt 模板
        prompt = prompt_template.format(context=context)
        
        try:
            options = {
                # "num_predict": num_predict,
                "temperature": temperature,
                'do_sample': True,
                # "top_p": top_p,
                # "top_k": top_k,
                # "repeat_penalty": repeat_penalty
            }
            # 将PIL Image对象转换为Ollama期望的格式（bytes）
            if isinstance(image, Image.Image):  # PIL Image对象
                from io import BytesIO
                buffer = BytesIO()
                image.save(buffer, format='JPEG')
                image_data = buffer.getvalue()
            elif isinstance(image, bytes):
                image_data = image
            else:
                # 其他格式，尝试转换为bytes
                image_data = image
            
            response = self.client.generate(
                model=self.model_name,
                # prompt=prompt[0]["content"][1]["text"],
                prompt=prompt,
                images=[image_data],
                options=options
            )
            ans = response["response"].strip()
            return ans, 128
        except Exception as e:
            return "**ERROR**: " + str(e), 0

    def chat(self, system, history, gen_conf, image=""):
        if system:
            history[-1]["content"] = system + history[-1]["content"] + "user query: " + history[-1]["content"]

        try:
            for his in history:
                if his["role"] == "user":
                    his["images"] = [image]
            options = {}
            if "temperature" in gen_conf: options["temperature"] = gen_conf["temperature"]
            if "max_tokens" in gen_conf: options["num_predict"] = gen_conf["max_tokens"]
            if "top_p" in gen_conf: options["top_k"] = gen_conf["top_p"]
            if "presence_penalty" in gen_conf: options["presence_penalty"] = gen_conf["presence_penalty"]
            if "frequency_penalty" in gen_conf: options["frequency_penalty"] = gen_conf["frequency_penalty"]
            response = self.client.chat(
                model=self.model_name,
                messages=history,
                options=options,
                keep_alive=-1
            )

            ans = response["message"]["content"].strip()
            return ans, response["eval_count"] + response.get("prompt_eval_count", 0)
        except Exception as e:
            return "**ERROR**: " + str(e), 0

    def chat_streamly(self, system, history, gen_conf, image=""):
        if system:
            history[-1]["content"] = system + history[-1]["content"] + "user query: " + history[-1]["content"]

        for his in history:
            if his["role"] == "user":
                his["images"] = [image]
        options = {}
        if "temperature" in gen_conf: options["temperature"] = gen_conf["temperature"]
        if "max_tokens" in gen_conf: options["num_predict"] = gen_conf["max_tokens"]
        if "top_p" in gen_conf: options["top_k"] = gen_conf["top_p"]
        if "presence_penalty" in gen_conf: options["presence_penalty"] = gen_conf["presence_penalty"]
        if "frequency_penalty" in gen_conf: options["frequency_penalty"] = gen_conf["frequency_penalty"]
        ans = ""
        try:
            response = self.client.chat(
                model=self.model_name,
                messages=history,
                stream=True,
                options=options,
                keep_alive=-1
            )
            for resp in response:
                if resp["done"]:
                    yield resp.get("prompt_eval_count", 0) + resp.get("eval_count", 0)
                ans += resp["message"]["content"]
                yield ans
        except Exception as e:
            yield ans + "\n**ERROR**: " + str(e)
        yield 0


class llamaCPPCV(Base):
    def __init__(self, key, model_name, lang="English", **kwargs):
        model_path = '/media/zzg/GJ_disk01/pretrained_model/bartowski/OpenGVLab_InternVL3_5-14B-GGUF/OpenGVLab_InternVL3_5-14B-Q5_K_M.gguf'
        clip_model_path = '/media/zzg/GJ_disk01/pretrained_model/bartowski/OpenGVLab_InternVL3_5-14B-GGUF/mmproj-OpenGVLab_InternVL3_5-14B-bf16.gguf'
        chat_handler = Llava15ChatHandler(clip_model_path=clip_model_path)
        self.model = Llama(
            model_path=model_path,
            chat_handler=chat_handler,
            n_ctx=4096,
            n_gpu_layers=-1,  # 所有模型层都尝试放到 GPU
            n_batch=256,
            verbose=False,
            n_threads=8,  # CPU線程數
        )

    def _preprocess_image_bytes(self, image, max_side=1024, quality=80):
        if isinstance(image, Image.Image):
            w,h = image.size
            scale = min(1.0, max_side / max(w,h))
            if scale < 1.0:
                image = image.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
            buf = BytesIO()
            image.save(buf, format='JPEG', quality=quality)
            return base64.b64encode(buf.getvalue()).decode()
        else:
            return image  # already bytes/base64

    def _build_prompt(self, context):
        prompt_template = '''You are analyzing an image that may contain charts, graphs, infographics, tables, maps, or other visual representations. 
        Your task is to provide an accurate, concise, and RAG-ready description focused on factual content suitable for a knowledge base.

        CONTEXT INTEGRATION:
        1. You have been provided with context from the PDF document: {context}
        2. FIRST, evaluate whether this context is relevant to the image content. 
        3. If the context is RELEVANT, integrate meaningful information from it to enhance your analysis
        4. If the context is IRRELEVANT, prioritize the actual image content and ignore the context

        SPECIFIC INSTRUCTIONS:
        1. CAREFULLY EXAMINE THE IMAGE CONTENT:
           - First determine the image type (financial chart, business graph, map, diagram, table, or other)
           - Adapt your analysis approach based on the image type

        2. FOR FINANCIAL/BUSINESS IMAGES (charts, graphs, tables):
           - First identify x-axis and y-axis meaning (e.g., x = year, y = monetary value). If y-axis ticks and unit are visible (e.g., "HK$M" and tick labels), include the unit in all reported values.
           - If the chart is stacked, attempt to extract values per legend item. If numeric labels are not present, perform linear interpolation between visible y-axis ticks and mark such values as "estimated".
           - Focus on numerical data: exact numbers, percentages, monetary values, dates and units.
           - Identify and report key financial/business metrics (revenue, profit, growth rates, market share, KPIs) only if clearly visible.
           - Highlight trends, comparisons, and significant changes using concise bullets.
           - Always report time periods, measurement units, and currency types when present.
           - For charts that use a legend (e.g., stacked bars or colored segments), map legend entries to visual segments by matching legend labels to visual indicators (prefer pixel/colour sampling, e.g., RGB/HEX) rather than vague color names. Use that mapping to assign each visible segment to a label/region.
           - If numeric labels are present on bars/segments/points, treat those as "measured". If numeric labels are absent but axis ticks and units are readable, convert pixel heights/positions to data values via linear interpolation between visible ticks.

        3. FOR NON-BUSINESS IMAGES (maps, diagrams, general visuals):
           - Provide a factual, concise description of the depicted elements and relationships.
           - Extract any visible text, labels, or numbers.
           - Report spatial or causal relationships if clearly shown.
           - Do not force a financial/business interpretation.

        4. DATA EXTRACTION PRINCIPLES:
           - Do not invent, assume, or extrapolate information that is not visibly present.
           - Report only clearly visible and readable information
           - Do not include decorative/visual attributes (color, shape, font, layout) unless they convey real data (e.g., legend mapping color → value). Descriptions like “blue oval” or “green box” are unnecessary for the knowledge base and must be omitted.

        5. OUTPUT FORMAT (RAG-FRIENDLY):
           - Produce a short, factual summary suitable for ingestion into a knowledge base.
           - Each bullet must be a single factual statement (no narrative fluff)
           - For data-rich images: use bullet points for key findings
           - Always maintain factual accuracy over speculative interpretation

        6. QUALITY CONTROL:
           - If the image is decorative or purely artistic without data, give one brief factual sentence (no numeric analysis).

        PRIMARY GOAL: produce a compact, factual, non-decorative description optimized for RAG ingestion — prioritize numbers and explicit claims; otherwise summarize clearly stated relationships without inventing details.
        '''

        # 使用 format 方法将 context 参数插入 prompt 模板
        prompt = prompt_template.format(context=context)
        return prompt

    def describe(self,
                 image,
                 context: str = '',
                 num_predict: int = 512,
                 temperature: float = 0.1,
                 top_p: float = 0.8,
                 top_k: int = 40,
                 repeat_penalty: float = 1.1):
        # b64 = self.image2base64(image)
        prompt = self._build_prompt(context)
        try:
            # 将PIL Image对象转换为Ollama期望的格式（bytes）
            img_base64 = self._preprocess_image_bytes(image, max_side=1024, quality=80)

                # 构建消息
            messages = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}}
                ]
            }]
            # 生成描述
            response = self.model.create_chat_completion(
                messages=messages,
                max_tokens=num_predict,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repeat_penalty=repeat_penalty,
            )
            ans = response["choices"][0]["message"]["content"]
            # 显式释放临时对象，调用 GC
            del img_base64, messages, response
            gc.collect()
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
            return ans, 128
        except Exception as e:
            gc.collect()
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
            return "**ERROR**: " + str(e), 0

    def chat(self, system, history, gen_conf, image=""):
        if system:
            history[-1]["content"] = system + history[-1]["content"] + "user query: " + history[-1]["content"]

        try:
            for his in history:
                if his["role"] == "user":
                    his["images"] = [image]
            options = {}
            if "temperature" in gen_conf: options["temperature"] = gen_conf["temperature"]
            if "max_tokens" in gen_conf: options["num_predict"] = gen_conf["max_tokens"]
            if "top_p" in gen_conf: options["top_k"] = gen_conf["top_p"]
            if "presence_penalty" in gen_conf: options["presence_penalty"] = gen_conf["presence_penalty"]
            if "frequency_penalty" in gen_conf: options["frequency_penalty"] = gen_conf["frequency_penalty"]
            response = self.client.chat(
                model=self.model_name,
                messages=history,
                options=options,
                keep_alive=-1
            )

            ans = response["message"]["content"].strip()
            return ans, response["eval_count"] + response.get("prompt_eval_count", 0)
        except Exception as e:
            return "**ERROR**: " + str(e), 0

    def chat_streamly(self, system, history, gen_conf, image=""):
        if system:
            history[-1]["content"] = system + history[-1]["content"] + "user query: " + history[-1]["content"]

        for his in history:
            if his["role"] == "user":
                his["images"] = [image]
        options = {}
        if "temperature" in gen_conf: options["temperature"] = gen_conf["temperature"]
        if "max_tokens" in gen_conf: options["num_predict"] = gen_conf["max_tokens"]
        if "top_p" in gen_conf: options["top_k"] = gen_conf["top_p"]
        if "presence_penalty" in gen_conf: options["presence_penalty"] = gen_conf["presence_penalty"]
        if "frequency_penalty" in gen_conf: options["frequency_penalty"] = gen_conf["frequency_penalty"]
        ans = ""
        try:
            response = self.client.chat(
                model=self.model_name,
                messages=history,
                stream=True,
                options=options,
                keep_alive=-1
            )
            for resp in response:
                if resp["done"]:
                    yield resp.get("prompt_eval_count", 0) + resp.get("eval_count", 0)
                ans += resp["message"]["content"]
                yield ans
        except Exception as e:
            yield ans + "\n**ERROR**: " + str(e)
        yield 0

class LocalAICV(GptV4):
    def __init__(self, key, model_name, base_url, lang="Chinese"):
        if not base_url:
            raise ValueError("Local cv model url cannot be None")
        if base_url.split("/")[-1] != "v1":
            base_url = os.path.join(base_url, "v1")
        self.client = OpenAI(api_key="empty", base_url=base_url)
        self.model_name = model_name.split("___")[0]
        self.lang = lang


class XinferenceCV(Base):
    def __init__(self, key, model_name="", lang="Chinese", base_url=""):
        if base_url.split("/")[-1] != "v1":
            base_url = os.path.join(base_url, "v1")
        self.client = OpenAI(api_key="xxx", base_url=base_url)
        self.model_name = model_name
        self.lang = lang

    def describe(self, image, max_tokens=300):
        b64 = self.image2base64(image)

        res = self.client.chat.completions.create(
            model=self.model_name,
            messages=self.prompt(b64),
            max_tokens=max_tokens,
        )
        return res.choices[0].message.content.strip(), res.usage.total_tokens

class GeminiCV(Base):
    def __init__(self, key, model_name="gemini-1.0-pro-vision-latest", lang="Chinese", **kwargs):
        from google.generativeai import client, GenerativeModel, GenerationConfig
        client.configure(api_key=key)
        _client = client.get_default_generative_client()
        self.model_name = model_name
        self.model = GenerativeModel(model_name=self.model_name)
        self.model._client = _client
        self.lang = lang 

    def describe(self, image, max_tokens=2048):
        from PIL.Image import open
        gen_config = {'max_output_tokens':max_tokens}
        prompt = "请用中文详细描述一下图中的内容，比如时间，地点，人物，事情，人物心情等，如果有数据请提取出数据。" if self.lang.lower() == "chinese" else \
            "Please describe the content of this picture, like where, when, who, what happen. If it has number data, please extract them out."
        b64 = self.image2base64(image) 
        img = open(BytesIO(base64.b64decode(b64))) 
        input = [prompt,img]
        res = self.model.generate_content(
            input,
            generation_config=gen_config,
        )
        return res.text,res.usage_metadata.total_token_count

    def chat(self, system, history, gen_conf, image=""):
        if system:
            history[-1]["content"] = system + history[-1]["content"] + "user query: " + history[-1]["content"]
        try:
            for his in history:
                if his["role"] == "assistant":
                    his["role"] = "model"
                    his["parts"] = [his["content"]]
                    his.pop("content")
                if his["role"] == "user":
                    his["parts"] = [his["content"]]
                    his.pop("content")
            history[-1]["parts"].append(f"data:image/jpeg;base64," + image)

            response = self.model.generate_content(history, generation_config=GenerationConfig(
                max_output_tokens=gen_conf.get("max_tokens", 1000), temperature=gen_conf.get("temperature", 0.3),
                top_p=gen_conf.get("top_p", 0.7)))

            ans = response.text
            return ans, response.usage_metadata.total_token_count
        except Exception as e:
            return "**ERROR**: " + str(e), 0

    def chat_streamly(self, system, history, gen_conf, image=""):
        if system:
            history[-1]["content"] = system + history[-1]["content"] + "user query: " + history[-1]["content"]

        ans = ""
        tk_count = 0
        try:
            for his in history:
                if his["role"] == "assistant":
                    his["role"] = "model"
                    his["parts"] = [his["content"]]
                    his.pop("content")
                if his["role"] == "user":
                    his["parts"] = [his["content"]]
                    his.pop("content")
            history[-1]["parts"].append(f"data:image/jpeg;base64," + image)

            response = self.model.generate_content(history, generation_config=GenerationConfig(
                max_output_tokens=gen_conf.get("max_tokens", 1000), temperature=gen_conf.get("temperature", 0.3),
                top_p=gen_conf.get("top_p", 0.7)), stream=True)

            for resp in response:
                if not resp.text: continue
                ans += resp.text
                yield ans
        except Exception as e:
            yield ans + "\n**ERROR**: " + str(e)

        yield response._chunks[-1].usage_metadata.total_token_count


class OpenRouterCV(GptV4):
    def __init__(
        self,
        key,
        model_name,
        lang="Chinese",
        base_url="https://openrouter.ai/api/v1",
    ):
        if not base_url:
            base_url = "https://openrouter.ai/api/v1"
        self.client = OpenAI(api_key=key, base_url=base_url)
        self.model_name = model_name
        self.lang = lang


class LocalCV(Base):
    def __init__(self, key, model_name="glm-4v", lang="Chinese", **kwargs):
        pass

    def describe(self, image, max_tokens=1024):
        return "", 0


class NvidiaCV(Base):
    def __init__(
        self,
        key,
        model_name,
        lang="Chinese",
        base_url="https://ai.api.nvidia.com/v1/vlm",
    ):
        if not base_url:
            base_url = ("https://ai.api.nvidia.com/v1/vlm",)
        self.lang = lang
        factory, llm_name = model_name.split("/")
        if factory != "liuhaotian":
            self.base_url = os.path.join(base_url, factory, llm_name)
        else:
            self.base_url = os.path.join(
                base_url, "community", llm_name.replace("-v1.6", "16")
            )
        self.key = key

    def describe(self, image, max_tokens=1024):
        b64 = self.image2base64(image)
        response = requests.post(
            url=self.base_url,
            headers={
                "accept": "application/json",
                "content-type": "application/json",
                "Authorization": f"Bearer {self.key}",
            },
            json={
                "messages": self.prompt(b64),
                "max_tokens": max_tokens,
            },
        )
        response = response.json()
        return (
            response["choices"][0]["message"]["content"].strip(),
            response["usage"]["total_tokens"],
        )

    def prompt(self, b64):
        return [
            {
                "role": "user",
                "content": (
                    "请用中文详细描述一下图中的内容，比如时间，地点，人物，事情，人物心情等，如果有数据请提取出数据。"
                    if self.lang.lower() == "chinese"
                    else "Please describe the content of this picture, like where, when, who, what happen. If it has number data, please extract them out."
                )
                + f' <img src="data:image/jpeg;base64,{b64}"/>',
            }
        ]

    def chat_prompt(self, text, b64):
        return [
            {
                "role": "user",
                "content": text + f' <img src="data:image/jpeg;base64,{b64}"/>',
            }
        ]


class StepFunCV(GptV4):
    def __init__(self, key, model_name="step-1v-8k", lang="Chinese", base_url="https://api.stepfun.com/v1"):
        if not base_url: base_url="https://api.stepfun.com/v1"
        self.client = OpenAI(api_key=key, base_url=base_url)
        self.model_name = model_name
        self.lang = lang


class LmStudioCV(GptV4):
    def __init__(self, key, model_name, lang="Chinese", base_url=""):
        if not base_url:
            raise ValueError("Local llm url cannot be None")
        if base_url.split("/")[-1] != "v1":
            base_url = os.path.join(base_url, "v1")
        self.client = OpenAI(api_key="lm-studio", base_url=base_url)
        self.model_name = model_name
        self.lang = lang


class OpenAI_APICV(GptV4):
    def __init__(self, key, model_name, lang="Chinese", base_url=""):
        if not base_url:
            raise ValueError("url cannot be None")
        if base_url.split("/")[-1] != "v1":
            base_url = os.path.join(base_url, "v1")
        self.client = OpenAI(api_key=key, base_url=base_url)
        self.model_name = model_name.split("___")[0]
        self.lang = lang


class TogetherAICV(GptV4):
    def __init__(self, key, model_name, lang="Chinese", base_url="https://api.together.xyz/v1"):
        if not base_url:
            base_url = "https://api.together.xyz/v1"
        super().__init__(key, model_name,lang,base_url)


class YiCV(GptV4):
    def __init__(self, key, model_name, lang="Chinese",base_url="https://api.lingyiwanwu.com/v1",):
        if not base_url:
            base_url = "https://api.lingyiwanwu.com/v1"
        super().__init__(key, model_name,lang,base_url)


class HunyuanCV(Base):
    def __init__(self, key, model_name, lang="Chinese",base_url=None):
        from tencentcloud.common import credential
        from tencentcloud.hunyuan.v20230901 import hunyuan_client

        key = json.loads(key)
        sid = key.get("hunyuan_sid", "")
        sk = key.get("hunyuan_sk", "")
        cred = credential.Credential(sid, sk)
        self.model_name = model_name
        self.client = hunyuan_client.HunyuanClient(cred, "")
        self.lang = lang

    def describe(self, image, max_tokens=4096):
        from tencentcloud.hunyuan.v20230901 import models
        from tencentcloud.common.exception.tencent_cloud_sdk_exception import (
            TencentCloudSDKException,
        )
        
        b64 = self.image2base64(image)
        req = models.ChatCompletionsRequest()
        params = {"Model": self.model_name, "Messages": self.prompt(b64)}
        req.from_json_string(json.dumps(params))
        ans = ""
        try:
            response = self.client.ChatCompletions(req)
            ans = response.Choices[0].Message.Content
            return ans, response.Usage.TotalTokens
        except TencentCloudSDKException as e:
            return ans + "\n**ERROR**: " + str(e), 0
        
    def prompt(self, b64):
        return [
            {
                "Role": "user",
                "Contents": [
                    {
                        "Type": "image_url",
                        "ImageUrl": {
                            "Url": f"data:image/jpeg;base64,{b64}"
                        },
                    },
                    {
                        "Type": "text",
                        "Text": "请用中文详细描述一下图中的内容，比如时间，地点，人物，事情，人物心情等，如果有数据请提取出数据。" if self.lang.lower() == "chinese" else
                        "Please describe the content of this picture, like where, when, who, what happen. If it has number data, please extract them out.",
                    },
                ],
            }
        ]