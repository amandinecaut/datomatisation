from google.generativeai import GenerationConfig
from typing import List, Union, Dict, Optional
from abc import ABC, abstractmethod
import google.generativeai as genai


import streamlit as st
import numpy as np
import pandas as pd
import re

import json
from scipy.stats import zscore



class Description(ABC):
    describe_base = "data/describe"

    #@property
    #@abstractmethod
    #def gpt_examples_path(self) -> str:
    #    """
    #    Path to excel files containing examples of user and assistant messages for the GPT to learn from.
    #    """


    @property
    @abstractmethod
    def describe_paths(self) -> Union[str, List[str]]:
        """
        List of paths to excel files containing questions and answers for the GPT to learn from.
        """

    def __init__(self):
        self.synthesized_text = self.synthesize_text()
        self.messages = self.setup_messages()

    def synthesize_text(self) -> str:
        """
        Return a data description that will be used to prompt GPT.

        Returns:
        str
        """

    def get_prompt_messages(self) -> List[Dict[str, str]]:
        """
        Return the prompt that the GPT will see before self.synthesized_text.

        Returns:
        List of dicts with keys "role" and "content".
        """

    def get_intro_messages(self) -> List[Dict[str, str]]:
        """
        Constant introduction messages for the assistant.

        Returns:
        List of dicts with keys "role" and "content".
        """
        intro = [
            {
                "role": "system",
                "content": (
                    "You are a data analysis bot. "
                    "You provide succinct and to the point explanations about data using data. "
                    "You use the information given to you from the data and answers "
                    "to earlier user/assistant pairs to give summaries of players."
                ),
            },
        ]
        if len(self.describe_paths) > 0:
            intro += [
                {
                    "role": "user",
                    "content": "First, could you answer some questions about the data for me?",
                },
                {"role": "assistant", "content": "Sure!"},
            ]

        return intro
        
    def get_messages_from_excel(self,paths: Union[str, List[str]],) -> List[Dict[str, str]]:
        """
        Turn an excel file containing user and assistant columns with str values into a list of dicts.

        Arguments:
        paths: str or list of str
            Path to the excel file containing the user and assistant columns.

        Returns:
        List of dicts with keys "role" and "content".

        """

        # Handle list and str paths arg
        if isinstance(paths, str):
            paths = [paths]
        elif len(paths) == 0:
            return []

        # Concatenate dfs read from paths
        df = pd.read_excel(paths[0])
        for path in paths[1:]:
            df = pd.concat([df, pd.read_excel(path)])

        if df.empty:
            return []

        # Convert to list of dicts
        messages = []
        for i, row in df.iterrows():
            if i == 0:
                messages.append({"role": "user", "content": row["user"]})
            else:
                messages.append({"role": "user", "content": row["user"]})
            messages.append({"role": "assistant", "content": row["assistant"]})

        return messages

    def setup_messages(self) -> List[Dict[str, str]]:
        messages = self.get_intro_messages()

        try:
            paths = self.describe_paths
            messages += self.get_messages_from_excel(paths)
        except (
            FileNotFoundError
        ) as e:  # FIXME: When merging with new_training, add the other exception
            print(e)
        messages += self.get_prompt_messages()

        messages = [
            message for message in messages if isinstance(message["content"], str)
        ]

        #try:
        #    messages += self.get_messages_from_excel(
        #        paths=self.gpt_examples_path,
        #    )
        #except (
        #    FileNotFoundError
        #) as e:  # FIXME: When merging with new_training, add the other exception
        #    print(e)
        
        messages += self.get_prompt_messages()

        messages = [
            message for message in messages if isinstance(message["content"], str)
        ]


        messages += [
            {
                "role": "user",
                "content": f"Now do the same thing with the following: ```{self.synthesized_text}```",
            }
        ]
        return messages

    def convert_messages_format(self, messages):
        new_messages = []
        system_prompt = None
        if len(messages) > 0 and messages[0]["role"] == "system":
            # If the first message is a system message, store it and return it.
            # Gemini requires the system prompt to be passed in separately.
            system_prompt = messages[0]["content"]
            messages = messages[1:]
        for message in messages:
            role = "model" if message["role"] == "assistant" else "user"
            new_message = {
                "role": role,
                "parts": message["content"],
            }
            new_messages.append(new_message)

        user_query = ""
        if new_messages[-1]["role"] == "user":
            user_query = new_messages.pop()
        return {"system_instruction": system_prompt, "history": new_messages, "content": user_query}

    def stream_gpt(self, temperature=1):
        """
        Run the GPT model on the messages and stream the output.

        Arguments:
        temperature: optional float
            The temperature of the GPT model.

        Yields:
            str
        """

        st.expander("Chat transcript", expanded=False).write(self.messages)

    
        msgs = self.convert_messages_format(self.messages)

        model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
            system_instruction=msgs["system_instruction"],
            generation_config=GenerationConfig(max_output_tokens=150),
            )
        chat = model.start_chat(history=msgs["history"])
        response = chat.send_message(
            content=msgs["content"],
        )

        answer = response.text

        return answer

    

class CreateDescription(Description):
    #output_token_limit = 150

    #@property
    #def gpt_examples_path(self):
    #    return f"{self.gpt_examples_base}/Forward.xlsx"

    @property
    def describe_paths(self):
        return [f"{self.describe_base}/QandA_data.xlsx"]


    def __init__(self):
        self.FA_df = st.session_state.FA_df
        self.FA_component_dict = st.session_state.FA_component_dict

        if st.session_state.selected_entity == None:
            self.indice = 0
        else:
            self.indice = st.session_state.df_filtered.index.tolist().index(
                st.session_state.selected_entity
            )
        super().__init__()

    def get_intro_messages(self) -> List[Dict[str, str]]:
        """
        Constant introduction messages for the assistant.

        Returns:
        List of dicts with keys "role" and "content".
        """
        intro = [
            {
                "role": "system",
                "content": (
                    "You are a data analyst. "
                    "You provide succinct and to the point explanations about the data.  "
                    "You use the information given to you from the data and answers"
                    "to earlier user/assistant pairs to give a description"
                ),
            }
        ]

        return intro

    def describe_level(self, value):
        thresholds=[-2, -1, -0.5, 0.5, 1, 2]
        words = [
            "is extremely low on ", 
            "is very low on ", 
            "is quite low on ", 
            "is relatively on ", 
            "is quite high on",  
            "is very high on ", 
            "is extremely high on "
            ]
        return self.describe(thresholds, words, value)

    def describe(self, thresholds, words, value):
        """
        thresholds = upper bound of each range in ascending order
        len(words) = len(thresholds) + 1
        """
        assert len(words) == len(thresholds) + 1, "Issue with thresholds and words"

        # Iterate through thresholds to find the correct description
    
        for i, threshold in enumerate(thresholds):
            if value < threshold:
                return words[i]

        # If no match (value exceeds the largest threshold), return the last word
        return words[-1]

    def get_description(self, indice):
        self.FA_df = st.session_state.FA_df.apply(zscore, nan_policy="omit")
        self.FA_df = self.FA_df.iloc[indice, :].to_frame().T

        text = ''
        for i in st.session_state.FA_component_dict.keys():
            component = st.session_state.FA_component_dict.get(i, {})
            if not component:  # Skip if component is missing
                continue
            
            text_left, text_right = self.split_qualities(component['label'])

            text += 'The entity '
        
            value = self.FA_df[i].values[0]
        
            if not np.isnan(value):
                if value >= 0:
                    text += self.describe_level(value) + text_right + '. '
                else:
                    text += self.describe_level(value) + text_left + '. '


                if value > 1:
                    text += 'In particular, the entity says that ' + component["top"][0] + '. '
                elif value < -1:
                    text += 'In particular, the entity says that ' + component["bottom"][0] + '. '
            

        return text

    def synthesize_text(self):

        description = self.get_description(self.indice)

        return description

    def get_prompt_messages(self):
        prompt = (
            f"Please use the statistical description enclosed with ``` to give a concise, 4 sentence summary of the entity. "
            f"The first sentence should use varied language to give an overview of the entity. "
            "The second sentence should describe the entity's specific strengths based on the metrics. "
            "The third sentence should describe aspects in which the entity is average and/or weak based on the statistics. "
            "Finally, summarise exactly how the entity compares to others in the same position. "
        )
        return [{"role": "user", "content": prompt}]

    def describe_level_cluster(self, value):
        thresholds=[-3,-2,-1.5, -1, -0.5, 0.5, 1,1.5, 2,3]
        words = [
        " extremely low on ",    
        " very low on ",         
        " quite low on ",       
        " relatively low on ",   
        " slightly low on ",     
        " normal on ",           
        " slightly high on ",    
        " relatively high on ", 
        " quite high on ",       
        " very high on ",       
        " extremely high on "  
        ]
        return self.describe(thresholds, words, value)
       
    def split_qualities(self, text):
        # Use a regular expression to split on " vs " (case-insensitive)
        parts = re.split(r"\s+vs\.?\s+", text, flags=re.IGNORECASE)

        # Ensure we have two sides
        if len(parts) != 2:
            raise ValueError("Text must contain 'vs' separating two sides.")

        text1, text2 = parts[0].strip(), parts[1].strip()

        return text1, text2

    def get_cluster_label(self, text):

        msgs = {
            "system_instruction": "You are a data analyst and scientist",
            "history": [
                {
                "role": "user",
                "parts": (
                    "You label a cluster."
                    "The label has to be short and smooth."
                    "Output a label only."
                    ),
                },
                ],
            "content": {"role": "user", "parts": text},
            }
        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            system_instruction=msgs["system_instruction"],
            generation_config=GenerationConfig(max_output_tokens=5),
            )
        chat = model.start_chat(history=msgs["history"])
        response = chat.send_message(
            content=msgs["content"],
            )
        return response.candidates[0].content.parts[0].text
    
    def get_cluster_description(self, text):

        msgs = {
            "system_instruction": "You are a data analyst and scientist",
            "history": [
                {
                "role": "user",
                "parts": (
                    "You have a description of a cluster. Make it better and more descriptive"
                    ),
                },
                ],
            "content": {"role": "user", "parts": text},
            }
        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            system_instruction=msgs["system_instruction"],
            generation_config=GenerationConfig(max_output_tokens=1000),
            )
        chat = model.start_chat(history=msgs["history"])
        response = chat.send_message(
            content=msgs["content"],
            )
        return response.candidates[0].content.parts[0].text


