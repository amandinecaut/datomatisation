from google.generativeai import GenerationConfig
from typing import List, Union, Dict, Optional
from abc import ABC, abstractmethod
import google.generativeai as genai
from scipy.stats import zscore
import streamlit as st
import numpy as np
import pandas as pd
import openai
import json
import toml
import re
import os





class Description(ABC):
    describe_base = "data/describe"

    @property
    @abstractmethod
    #def gpt_examples_path(self) -> str:
    #    """
    #    Path to excel files containing examples of user and assistant messages for the GPT to learn from.
    #    """



    def describe_paths(self) -> Union[str, List[str]]:
        """
        List of paths to excel files containing questions and answers for the GPT to learn from.
        """

    def __init__(self):
        self.synthesized_text = self.synthesize_text()
        self.messages = self.setup_messages()
        self._config = toml.load(".streamlit/secrets.toml")

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
                    "Use earlier user/assistant pairs to give summaries of the data"
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

        _, ext = os.path.splitext(paths[0])
        ext = ext.lower()

        if ext == ".csv":
            df = pd.read_csv(paths[0])
            for path in paths[1:]:
                df = pd.concat([df, pd.read_csv(path)])

        elif ext in [".xls", ".xlsx"]:
            df = pd.read_excel(paths[0])
            for path in paths[1:]:
                df = pd.concat([df, pd.read_excel(path)])
        else:
            raise ValueError(f"Unsupported file extension: {ext}")


        if df.empty:
            return []

        # Convert to list of dicts
        messages = []
        for i, row in df.iterrows():
            if i == 0:
                messages.append({"role": "user", "content": row["User"]})
            else:
                messages.append({"role": "user", "content": row["User"]})
            messages.append({"role": "assistant", "content": row["Assistant"]})

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
       
        answer = self.MH.get_generate(msgs, 150)
       

        return answer

    

class CreateDescription(Description):


    @property
    def describe_paths(self):
        return [f"{self.describe_base}/QandA_data.csv"]


    def __init__(self):
        if "FA_df" not in st.session_state:
            st.session_state["FA_df"] = pd.DataFrame()  
        if "FA_component_dict" not in st.session_state:
            st.session_state["FA_component_dict"] = {}


        self.FA_df = st.session_state.FA_df
        self.FA_component_dict = st.session_state.FA_component_dict

        if st.session_state.selected_entity == None:
            self.indice = 0
        else:
            self.indice = st.session_state.df_filtered.index.tolist().index(
                st.session_state.selected_entity
            )
        self.MH = ModelHandler()
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
                    "The label has to be short and clear."
                    "The label should not have connotation negative."
                    "The label has to be different from previous labels."
                    "Output a label only."
                    ),
                },
                ],
            "content": {"role": "user", "parts": text},
            }
        text_generate = self.MH.get_generate(msgs, max_output_token = 5)
        return text_generate #.candidates[0].content.parts[0].text
    
    def get_cluster_description(self, text):

        msgs = {
            "system_instruction": "You are a data analyst and scientist",
            "history": [
                {
                "role": "user",
                "parts": (
                    "You have a description of a cluster. Make it better and more descriptive. Give only one option"
                    ),
                },
                ],
            "content": {"role": "user", "parts": text},
            }
        text_generate = self.MH.get_generate(msgs,max_output_token = 500)
        
        return text_generate 


class ModelHandler:
    def __init__(self, config_path=".streamlit/secrets.toml"):
        self._config = toml.load(config_path)

    def get_model(self):
        """
        Returns a list of model tuples (model_object, service_name) in order of preference.
        """
        _config = toml.load(".streamlit/secrets.toml")
        models = []
    
        # Try to initialize Gemini model
        if _config["settings"].get("USE_GEMINI", True):
            try:
                config = _config["services"]["gemini"]
                genai.configure(api_key=config["GEMINI_API_KEY"])
                model = genai.GenerativeModel(
                    model_name=config["GEMINI_CHAT_MODEL"],
                )
                models.append((model, "gemini"))
            except Exception as e:
                print(f"Failed to initialize Gemini model: {e}")

        # Try to initialize GPT model
        try:
            config = _config["services"]["gpt"]
            model = "gpt-4o-mini"
    
            models.append((model, "gpt"))
        except Exception as e:
            print(f"Failed to initialize GPT model: {e}")
        
        return models

    def get_generate(self, msgs, max_output_token):
        """
        Attempts to use the primary model, and falls back to the next available model
        in case of a 429 quota error.
        """
        available_models = self.get_model()

        if not available_models:
            raise RuntimeError("No models could be initialized. Please check your configuration.")

        for model, service_name in available_models:
            try:
                print(f"Attempting to use {service_name.capitalize()} model...")
                
                
                # This is where you would make the actual API call
                if service_name == "gemini":
                    config = genai.GenerationConfig(max_output_tokens=max_output_token)
                    sys_instr, hist, user_msg= self.transform_msgs_for_gemini(msgs)
                    chat = model.start_chat(history=hist)
                    response = chat.send_message(user_msg)

                    response = response.candidates[0].content.parts[0].text
                    
                elif service_name == "gpt":
                    config = self._config["services"]["gpt"]
                    openai.api_key = config.get("GPT_KEY")
                    openai.api_base = config.get("GPT_BASE")  
                    openai.api_type = "azure"
                    openai.api_version = config.get("GPT_VERSION")

                    msgs = self.transform_msgs_for_azure(msgs)
    
                    # deployment_id must match your Azure deployment name
                    response_obj = openai.ChatCompletion.create(
                        deployment_id=model,  
                        messages=msgs,
                        temperature=1,
                        )       
                    response = response_obj.choices[0].message["content"].strip()
              
                return response

            except Exception as e:
                error_str = str(e)
                if "ResourceExhausted" in error_str or "429" in error_str:
                    print(f"{service_name.capitalize()} quota exceeded (429). Trying fallback...")
                    continue  # Try the next model in the list
                else:
                    raise  # Re-raise other errors

    def transform_msgs_for_azure(self, msgs):
        """
        Transform custom message structure into a list of messages compatible
        with Azure OpenAI ChatCompletion API.
        """
        valid_roles = {"system", "assistant", "user", "function", "tool", "developer"}
        azure_messages = []

        # --- If it's already a list of Azure-style messages, just return it ---
        if isinstance(msgs, list):
            return msgs

        # Add system instruction as first message if present
        system_instruction = msgs.get("system_instruction")
        if system_instruction:
            azure_messages.append({"role": "system", "content": str(system_instruction)})

        # Process history
        for msg in msgs.get("history", []):
            role = msg.get("role")
            if role not in valid_roles:
                continue  # skip invalid roles
            parts = msg.get("parts")
            if isinstance(parts, (list, tuple)):
                content = " ".join(parts)
            else:
                content = str(parts)
            azure_messages.append({"role": role, "content": content})

        # Add current content
        content_msg = msgs.get("content")
        if content_msg:
            role = content_msg.get("role", "user")
            if role not in valid_roles:
                role = "user"
            parts = content_msg.get("parts")
            if isinstance(parts, (list, tuple)):
                content = " ".join(parts)
            else:
                content = str(parts)
            azure_messages.append({"role": role, "content": content})

        return azure_messages
    def transform_msgs_for_gemini(self, msgs):
        """
        Transform custom message structure into a list of messages compatible
        with Gemini API.
        """
        # If msgs is already a list of GPT-like messages,
        #    convert *that* to Gemini format.
        if isinstance(msgs, list):
            system_instruction = None
            gemini_history = []
            for m in msgs:
                role = m.get("role")
                if role == "system":
                    system_instruction = m.get("content")
                elif role == "assistant":
                    gemini_history.append({"role": "model", "parts": [m.get("content", "")]})
                elif role == "user":
                    gemini_history.append({"role": "user", "parts": [m.get("content", "")]})
            # last user message is the new query
            user_message = None
            if gemini_history and gemini_history[-1]["role"] == "user":
                user_message = gemini_history[-1]["parts"][0]
            return system_instruction, gemini_history, user_message
        
        # Otherwise assume it's the custom dict shape
        elif isinstance(msgs, dict):
            system_instruction = msgs.get("system_instruction")
            gemini_history = []
            for msg in msgs.get("history", []):
                role = msg.get("role")
                if role == "assistant":
                    role = "model"
                elif role not in ("user", "model"):
                    continue
                parts = msg.get("parts")
                text = " ".join(parts) if isinstance(parts, (list, tuple)) else str(parts)
                gemini_history.append({"role": role, "parts": [text]})
            user_msg_obj = msgs.get("content")
            if user_msg_obj:
                parts = user_msg_obj.get("parts")
                user_message = " ".join(parts) if isinstance(parts, (list, tuple)) else str(parts)
            else:
                user_message = None
            return system_instruction, gemini_history, user_message

        else:
            raise TypeError(f"Unexpected msgs type: {type(msgs)}")