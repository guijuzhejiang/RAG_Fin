# -*- coding: utf-8 -*-
# The following documents are mainly referenced, and only adaptation modifications have been made
# from https://github.com/langchain-ai/langchain/blob/master/libs/text-splitters/langchain_text_splitters/json.py

import json
from typing import Any, Dict, List, Optional
from rag.nlp import find_codec

class RAGFlowJsonParser:
    def __init__(
        self, max_chunk_size: int = 2000, min_chunk_size: Optional[int] = None
    ):
        super().__init__()
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = (
            min_chunk_size
            if min_chunk_size is not None
            else max(int(max_chunk_size/4), 50)
        )

    def __call__(self, binary):
        encoding = find_codec(binary)
        txt = binary.decode(encoding, errors="ignore")
        json_data = json.loads(txt)
        chunks = self.split_json(json_data, True)   
        sections = [json.dumps(l, ensure_ascii=False) for l in chunks if l]
        return sections

    @staticmethod
    def _json_size(data: Dict) -> int:
        """Calculate the size of the serialized JSON object."""
        return len(json.dumps(data, ensure_ascii=False))

    @staticmethod
    def _set_nested_dict(d: Dict, path: List[str], value: Any) -> None:
        """Set a value in a nested dictionary based on the given path."""
        for key in path[:-1]:
            d = d.setdefault(key, {})
        d[path[-1]] = value

    def _list_to_dict_preprocessing(self, data: Any) -> Any:
        if isinstance(data, dict):
            # Process each key-value pair in the dictionary
            return {k: self._list_to_dict_preprocessing(v) for k, v in data.items()}
        elif isinstance(data, list):
            # Convert the list to a dictionary with index-based keys
            # return {
            #     str(i): self._list_to_dict_preprocessing(item)
            #     for i, item in enumerate(data)
            # }
            # 对于列表，仅递归处理子项，返回链接起来的字符串，而不转化为字典
            return '\n'.join(self._list_to_dict_preprocessing(item) for item in data)
        else:
            # Base case: the item is neither a dict nor a list, so return it unchanged
            return data
        
    def _json_split(self, data: Dict[str, Any], chunks: Optional[List[Dict]] = None) -> List[Dict]:
        """
        Split json into maximum size dictionaries while preserving structure.
        """
        chunks = chunks or []
        if isinstance(data, dict):
            for key, value in data.items():
                size = self._json_size({key: value})

                if size > self.max_chunk_size:
                    # Value exceeds max_chunk_size, split it
                    split_values = self._split_by_punctuation(value)
                    for i, split_value in enumerate(split_values):
                        # Add each split chunk separately
                        chunks.append({key: split_value})
                else:
                    # Add the key-value pair as a separate chunk
                    chunks.append({key: value})
        else:
            print("Invalid data type:", type(data), data)
        return chunks

    def split_json(
        self,
        json_data: Dict[str, Any],
        convert_lists: bool = False,
    ) -> List[Dict]:
        """Splits JSON into a list of JSON chunks"""

        if convert_lists:
            chunks = self._json_split(self._list_to_dict_preprocessing(json_data))
        else:
            chunks = self._json_split(json_data)

        # Remove the last chunk if it's empty
        if not chunks[-1]:
            chunks.pop()
        return chunks

    def split_text(
        self,
        json_data: Dict[str, Any],
        convert_lists: bool = False,
        ensure_ascii: bool = True,
    ) -> List[str]:
        """Splits JSON into a list of JSON formatted strings"""

        chunks = self.split_json(json_data=json_data, convert_lists=convert_lists)

        # Convert to string
        return [json.dumps(chunk, ensure_ascii=ensure_ascii) for chunk in chunks]

    def _split_by_punctuation(self, value: str) -> List[str]:
        """
        Split a string into smaller chunks at the nearest punctuation while ensuring each chunk is within max_size.
        """
        import re
        if not isinstance(value, str):
            print("Invalid value type:", type(value), value)
            raise ValueError("Only string values can be split by punctuation.")

        # Regular expression to find punctuation
        punctuation = re.compile(r'[,.!?;、。！？；]')
        chunks = []
        start = 0

        while start < len(value):
            # Try to make the current chunk as large as possible, up to max_chunk_size
            end = min(start + self.max_chunk_size, len(value))

            # If the remaining text is small, add it as the last chunk
            if len(value) - start <= self.max_chunk_size:
                chunks.append(value[start:])
                break

            # Look for the nearest punctuation from the end of the range back to start
            split_point = None
            for i in range(end, start, -1):
                if punctuation.match(value[i - 1]):
                    split_point = i
                    break

            # If no punctuation is found, split at max_chunk_size
            if split_point is None:
                split_point = end

            # Add the chunk and move the start pointer
            chunks.append(value[start:split_point])
            start = split_point

        return chunks
