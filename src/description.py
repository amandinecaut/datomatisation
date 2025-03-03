from google.generativeai import GenerationConfig
from typing import List, Union, Dict, Optional
from abc import ABC, abstractmethod
import google.generativeai as genai


import streamlit as st
import numpy as np
import pandas as pd

import json
from scipy.stats import zscore




class Description(ABC):

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

        return intro


    def setup_messages(self) -> List[Dict[str, str]]:
        messages = self.get_intro_messages()
        
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

    def get_1_label(self, text):

        msgs = {
            "system_instruction": "You are a data analyst and scientist",
            "history": [
                {
                "role": "user",
                "parts": (
                    "Determine in one word the subject."
                    "The will be x vs y, you ouput only 1 word from it. "
                    "Output a label only."),
                },
                {"role": "model", "parts": "Sure!"},
                ],
            "content": {"role": "user", "parts": text},
            }
        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            system_instruction=msgs["system_instruction"],
            generation_config=GenerationConfig(max_output_tokens=150),
            )
        chat = model.start_chat(history=msgs["history"])
        response = chat.send_message(
            content=msgs["content"],
            )
        return response.candidates[0].content.parts[0].text

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

            label = self.get_1_label(component.get("label", ""))

            text += 'The entity '
        
            value = self.FA_df[i].values[0]
        
            if not np.isnan(value):
                text += self.describe_level(value) + label + '. '

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
