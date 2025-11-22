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




class Wordalisation(ABC):
    describe_base = "data/describe"

    @property
    @abstractmethod
    def tell_it_how_to_answer(self) -> str:
        """
        Path to excel files containing examples of user and assistant messages for the few-shot prompting to learn from.
        """

    @property
    @abstractmethod
    def tell_it_what_it_knows(self) -> Union[str, List[str]]:
        """
        List of paths to excel files containing questions and answers for the injection of knowledge.
        """

    def __init__(self):
        #self.synthetic_text = self.tell_it_what_data_to_use()
        #self.messages = self.setup_messages()
        self._config = toml.load(".streamlit/secrets.toml")
        
    def tell_it_what_data_to_use(self) -> str:
        """
        Return a data description that will be used to prompt.

        Returns:
        str
        """

    def get_prompt_messages(self) -> List[Dict[str, str]]:
        """
        Return the prompt that the LLM  will see before self.synthetic_text.

        Returns:
        List of dicts with keys "role" and "content".
        """

    def tell_it_who_it_is(self) -> List[Dict[str, str]]:
        """
        Constant introduction messages for the assistant.

        Returns:
        List of dicts with keys "role" and "content".
        """
  
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
        """
        Return a data description that will be used to prompt the LLM.

        Returns:
        list of dicts with keys "role" and "content".
        """

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
        Run the LLM model on the messages and stream the output.

        Arguments:
        temperature: optional float
            The temperature of the LLM model.

        Yields:
            str
        """
        
        st.expander("Chat transcript", expanded=False).write(self.messages)
    
        msgs = self.convert_messages_format(self.messages)
        
        MH = ModelHandler()
        answer = MH.get_generate(msgs, 150)
       

        return answer

class CreateWordalisation(Wordalisation):

    @property
    def tell_it_how_to_answer(self):
        return f"{self.describe_base}/few_shot/few_shot_examples.xlsx"

    @property
    def tell_it_what_it_knows(self):
        return [f"{self.describe_base}/generate/QandA_data.csv"]


    def __init__(self):
        if "df" not in st.session_state:
            st.session_state["df"] = pd.DataFrame()  
        if "FA_component_dict" not in st.session_state:
            st.session_state["FA_component_dict"] = {}


        self.df = st.session_state.df
        self.FA_component_dict = st.session_state.FA_component_dict
        
        self.indice = st.session_state.indice
        self.entity_id = st.session_state.entity_id
        self.synthetic_text = self.tell_it_what_data_to_use()
        self.messages = self.setup_messages()
        
        
        
        super().__init__()

    def tell_it_who_it_is(self) -> List[Dict[str, str]]:
        """
        Constant introduction messages for the assistant.

        Returns:
        List of dicts with keys "role" and "content".
        """
        intro = [
            {
                "role": "system",
                "content": (
                    "You are an expert in interpreting and summarizing the results of complex statistical analyses. \n"
                    f"You have recently conducted a factor analysis in which you classified a set of {self.entity_id}, in multiple dimensions and grouped them based on shared characteristics. \n"
                    f"Your current task is to describe a specific {self.entity_id} in the context of this classification. \n"
                    "Before doing so, you will answer a series of questions related to the underlying scales and clusters."
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
            "is quite high on ",  
            "is very high on ", 
            "is extremely high on "
            ]
        return CreateWordalisation.describe(thresholds, words, value)

    @staticmethod
    def describe(thresholds, words, value):
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
        df = self.df[list(self.FA_component_dict.keys())]
        #df = df.apply(zscore, nan_policy="omit") # already done after the factor analysis in the perform_fa function
        indice = self.indice

        dictionary = st.session_state.col_mapping
        results_dict = st.session_state.FA_component_dict

        text = ''

        for component_key, component in results_dict.items():
            if not component:
                continue

            text_left, text_right = ClusterWordalisation.split_qualities(component['label'])
            text += f"{st.session_state.selected_entity} "
            value = df.loc[indice, component_key]

            if np.isnan(value):
                continue

            # Describe the value
            if value >= 0:
                text += self.describe_level(value) + text_right + '. '
            else:
                text += self.describe_level(value) + text_left + '. '

            # Map top features to columns
            top_map_list = [
             k for k, v in dictionary.items() if v in component["top"]
            ]
            
            bottom_map_list = [
             k for k, v in dictionary.items() if v in component["bottom"]
            ]

            # Fast argmax / argmin using NumPy
            if value > 1 and top_map_list:
                top_values = st.session_state.df_filtered.loc[indice, top_map_list].values
                argmax_idx = np.argmax(top_values)
                argmax_column = top_map_list[argmax_idx]
                text += f"In particular, {st.session_state.selected_entity} indicates that {dictionary[argmax_column]}. "

            elif value < -1 and bottom_map_list:
                bottom_values = st.session_state.df_filtered.loc[indice, bottom_map_list].values
                argmin_idx = np.argmin(bottom_values)
                argmin_column = bottom_map_list[argmin_idx]
                text += f"In particular, {st.session_state.selected_entity} indicates that {dictionary[argmin_column]}. "

        return text

    def get_description_cluster_entity(self):
        indice = st.session_state.indice
        df = self.df
        #cluster_number = int(df.at[indice, 'Cluster'])
        cluster_number = int(df.loc[indice, 'Cluster'])
        cluster_name = st.session_state.list_cluster_name[cluster_number]
        cluster_desc = st.session_state.list_description_cluster[cluster_number]
        entity = st.session_state.selected_entity

        text = f"{entity} is {cluster_name}. The people that are {cluster_name} has the following description: {cluster_desc}."

        return text

    def tell_it_what_data_to_use(self):

        self.synthetic_text = self.get_description(self.indice)
        return self.synthetic_text 

    def get_prompt_messages(self):
        prompt = (
            f"You will describe a specific {self.entity_id} based on its statistical description and informative description.\n"
            f"{self.get_description_cluster_entity()}\n"
            f"Please use the statistical description enclosed with ``` to give a concise, four sentence summary of the {self.entity_id}. \n"
            f"The first sentence should use varied language to give an overview of the {self.entity_id}. \n"
            f"The second sentence should describe the {self.entity_id}'s specific strengths based on the metrics. \n"
            f"The third sentence should describe aspects in which the {self.entity_id} is average and/or weak based on the statistics. \n"
            f"Finally, summarize the {self.entity_id} with a single concluding statement. \n" 
            f"Here is the statistical description of the {self.entity_id}: ```{self.synthetic_text}```"
        )
        return [{"role": "user", "content": prompt}]

    def tell_it_what_it_knows_cluster(self):
        cluster_name = st.session_state.list_cluster_name
        cluster_desc = st.session_state.list_description_cluster

    
        messages = [
            {
                "role": "user",
                "content": "You will be provided the background informations"
            },
            {
                "role": "assistant",
                "content": f"Understood. I will use this information when describing {self.entity_id}."
            },
            
        ]
        for name, desc in zip(cluster_name, cluster_desc):
            messages.extend([
                {
                    "role": "user",
                    "content": f"What is '{name}' about?"
                },
                {
                    "role": "assistant",
                    "content": f" '{name}' is described as: {desc}"
                }
            ])


        return messages

    def setup_messages(self) -> List[Dict[str, str]]:
        """Builds and returns a list of chat messages for model input."""
        messages = self.tell_it_who_it_is()

        # --- Load QandA ---
        try:
            tell_it_what_it_knows_paths = self.tell_it_what_it_knows
            messages += self.get_messages_from_excel(tell_it_what_it_knows_paths)
        except FileNotFoundError as e:
            # FIXME: When merging with new_training, add the other exception type
            print(f"Describe paths file not found: {e}")

        # --- Load Clusters description ---
        messages += self.tell_it_what_it_knows_cluster()
        

        # --- Filter out non-string content ---
        messages = [m for m in messages if isinstance(m.get("content"), str)]

        # --- Load few-shots examples  ---
        messages += [{
            "role": "user",
            "content": (
            f"Your task is to summarize a specific {self.entity_id}.\n"
            f"You will be provided with descriptions of {self.entity_id} from previous analyses on different datasets.\n"
            f"These examples illustrate the type of language you use and how you describe {self.entity_id} in terms of scales and clusters.\n"
            f"For each {self.entity_id}, provide a concise four sentence summary.\n"
            f"The first sentence should use varied language to give an overview of the {self.entity_id}. \n"
            f"The second sentence should describe the {self.entity_id}'s specific strengths based on the metrics. \n"
            f"The third sentence should describe aspects in which the {self.entity_id} is average and/or weak based on the statistics. \n"
            f"Finally, summarize the {self.entity_id} with a single concluding statement. \n" 
            )
            },
            {"role": "assistant",
            "content": f"Understood. Please provide the {self.entity_id} descriptions."
            }]

        try:
            example_paths = self.tell_it_how_to_answer
            messages += self.get_messages_from_excel(example_paths)
        except FileNotFoundError as e:
            # FIXME: When merging with new_training, add the other exception type
            print(f"Example paths file not found: {e}")

        # --- Add prompt messages ---
        messages += self.get_prompt_messages()


        return messages

class ClusterWordalisation(Wordalisation):

    @property
    def tell_it_how_to_answer(self):
        return [f"{self.describe_base}/few_shot/few_shot_cluster_description.xlsx"] 

    @property
    def tell_it_what_it_knows(self):
        return [f"{self.describe_base}/generate/QandA_data.csv"]

    def __init__(self):
        self.MH = ModelHandler()
        super().__init__()

    def describe_level_cluster(self, value):
        thresholds=[-2,-1.5, -1, -0.5,-0.25,0.25, 0.5, 1,1.5, 2]
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
        return CreateWordalisation.describe(thresholds, words, value)

    @staticmethod   
    def split_qualities(text):
        # Use a regular expression to split on " vs " (case-insensitive)
        parts = re.split(r"\s+vs\.?\s+", text, flags=re.IGNORECASE)

        # Ensure we have two sides
        if len(parts) != 2:
            raise ValueError("Text must contain 'vs' separating two sides.")

        text1, text2 = parts[0].strip(), parts[1].strip()

        return text1, text2


    def tell_it_who_it_is(self) -> List[Dict[str, str]]:
        """
        Constant introduction messages for the assistant.

        Returns:
        List of dicts with keys "role" and "content".
        """
        intro = [
            {
                "role": "system",
                "content": (
                    "You are a data analyst and you did a factor analysis, then a clustering. \n"
                    "You are task is to describe some clusters given the provided informations. \n"
                    "You will get a description of each cluster based on the latent factors. \n"
                    "First, you will be provided with a set of questions and answers that give you the necessary informations on each factors."
                ),
            },
            {
                "role": "assistant",
                "content": (    
                    "Understood. I will follow your instructions to describe the clusters based on the provided information."
                ),
            }
        ]

        return intro

    def get_prompt_messages(self):
        prompt = (
            "Now your task is to provide a description of a cluster.\n"
            "You will receive information about the cluster based on the latent factors that best describe it.\n"
            "For each cluster, write a concise summary based on the available information.\n"
            "The first sentence should give an overview of the cluster. \n"
            "The second sentence should describe the cluster’s specific strengths based on the available information.\n"
            "The third sentence should highlight areas where the cluster has specific weaknesses based on the available information.\n"
            f"Now do the same thing with the following: ```{self.synthetic_text}```"
        )
        
        return [{"role": "user", "content": prompt}]

    def description_cluster(self, center): 
        list_name_dim = []
        for _ , details in st.session_state.FA_component_dict.items():
            list_name_dim.append(details['label'])
        
        list_description_cluster = []

        
        describe_center = []
        for dim in np.arange(len(center)):
            value_dim = center[dim]
            text_dim = self.describe_level_cluster(value_dim)
            text_low, text_high = self.split_qualities(list_name_dim[dim])

            if value_dim >= 0:
                text_dim += text_high
            else:
                text_dim += text_low
            describe_center.append(text_dim)

        text = ", ".join(describe_center)  
        full_text = f"Members of this cluster can be described as: {text}"
            
        return full_text

    def tell_it_what_data_to_use(self, center):
        description = self.description_cluster(center)
        self.synthetic_text = description

        return self.synthetic_text 

    def setup_messages(self) -> List[Dict[str, str]]:
        """Builds and returns a list of chat messages for model input."""

        # ---- Tell it who it is ----
        messages = self.tell_it_who_it_is()

        # ---- Tell it what it knows ----
        #messages += self.get_messages_from_excel([f"{self.describe_base}/generate/tell_it_what_it_knows.csv"])
        # --- Load QandA ---
        try:
            tell_it_what_it_knows_paths = self.tell_it_what_it_knows
            messages += self.get_messages_from_excel(tell_it_what_it_knows_paths)
        except FileNotFoundError as e:
            # FIXME: When merging with new_training, add the other exception type
            print(f"Describe paths file not found: {e}")

        # -----Tell it how to answer ----
        # --- Load few-shots examples  ---
        messages += [{
            "role": "user",
            "content": (
                "Now your task is to provide a description of a cluster.\n"
                "You will receive information about the cluster based on the latent factors that best describe it.\n"
                "For each cluster, write a concise summary based on the available information.\n"
                "The first sentence should give an overview of the cluster. \n"
                "The second sentence should describe the cluster’s specific strengths based on the available information.\n"
                "The third sentence should highlight areas where the cluster has specific weaknesses based on the available information.\n"
                "I will provided you with the cluster description and you will return the summary. "
            )   
            },
            {"role": "assistant",
            "content": "Understood. Please provide the cluster descriptions."
            }]


        try:
            example_paths = self.tell_it_how_to_answer
            messages += self.get_messages_from_excel(example_paths)
        except FileNotFoundError as e:
            # FIXME: When merging with new_training, add the other exception type
            print(f"Example paths file not found: {e}")
        
        # ---- Tell it what data to use ----
        # --- Add prompt messages ---
        messages += self.get_prompt_messages()



        return messages

class Clusterlabel(Wordalisation):
    @property
    def tell_it_how_to_answer(self):
        return [f"{self.describe_base}/few_shot/few_shot_cluster_label.xlsx"] 

    @property
    def tell_it_what_it_knows(self):
        return [f"{self.describe_base}/generate/QandA_data.csv"]
    
    def __init__(self):
        self.MH = ModelHandler()
        super().__init__()

    def tell_it_who_it_is(self) -> List[Dict[str, str]]:
        """
        Constant introduction messages for the assistant.

        Returns:
        List of dicts with keys "role" and "content".
        """
        intro = [
            {
                "role": "system",
                "content": (
                    "You are a data analyst. \n"
                    "You are going to label clusters. \n"
                    "First, you will be provided with a set of questions and answers that give you the necessary context."
                ),
            }
        ]

        return intro

    def get_prompt_messages(self):
        prompt = (
            "Generate a short label for the clusters.\n" 
            "The label is maximum 2 words.\n" 
            "The label must have a positive or neutral connotation. The label should not have negative connotation.\n" 
            "Output a label only — nothing else.\n"
            f"{self.existing_labels_text}\n"
            f"Now do the same thing with the following: ```{self.synthetic_text}```"
        )
        
        return [{"role": "user", "content": prompt}]
    
    def tell_it_what_data_to_use(self, cluster_description):
        self.synthetic_text = cluster_description
        return self.synthetic_text 

    def existing_labels(self, list_labels):
        if not list_labels:
            self.existing_labels_text = ''
            return self.existing_labels_text
        elif len(list_labels) ==1:
            self.existing_labels_text = "The existing label is: " + list_labels[0] + ". The label must be different from previous label."
        else:
            self.existing_labels_text = "The existing labels are: " + ", ".join(list_labels) + ". The label must be different from previous labels."
        return self.existing_labels_text

    def setup_messages(self) -> List[Dict[str, str]]:
        """Builds and returns a list of chat messages for model input."""
        messages = self.tell_it_who_it_is()

        # --- Load QandA ---
        try:
            tell_it_what_it_knows_paths = self.tell_it_what_it_knows
            messages += self.get_messages_from_excel(tell_it_what_it_knows_paths)
        except FileNotFoundError as e:
            # FIXME: When merging with new_training, add the other exception type
            print(f"Describe paths file not found: {e}")

        # --- Load few-shots examples  ---
        messages += [{
            "role": "user",
            "content": (
                "Generate a short label for the clusters.\n" 
                "The label is maximum 2 words.\n" 
                "The label must have a positive or neutral connotation. The label should not have negative connotation.\n" 
                "The label must be different from previous labels.\n"
                "Output a label only — nothing else.\n"
                "I will provide you with cluster descriptions and you will return the label in the format specified an nothing else. "
            )}, 
            {
                "role": "assistant",
                "content": "Understood. Please provide the first cluster description."
            }]

        try:
            example_paths = self.tell_it_how_to_answer
            messages += self.get_messages_from_excel(example_paths)
        except FileNotFoundError as e:
            # FIXME: When merging with new_training, add the other exception type
            print(f"Example paths file not found: {e}")

        # --- Add prompt messages ---
        messages += self.get_prompt_messages()

        return messages


class FA(Wordalisation):
    # Functions used in both labelling and describing labels.

    def tell_it_what_data_to_use(self, FA_component_dict):

        self.synthetic_text = self.description_FA(FA_component_dict)
        return self.synthetic_text 

    def describe_level_FA(self, value):
        thresholds=[0.30, 0.49, 0.59, 0.70]
        if value > 0: 
            text = 'positively'
        else: 
            text = 'negatively'

        words = [
        f"very weakly {text} associated with ",  
        f"weakly {text} associated with ",  
        f"moderately {text} associated with ",
        f"strongly {text} associated with ",
        f"very strongly {text} associated with ", 
        ]
        return CreateWordalisation.describe(thresholds, words, abs(value))

    def existing_labels(self, list_labels):
        if not list_labels:
            self.existing_labels_text = ''
        elif len(list_labels) ==1:
            self.existing_labels_text = "An existing name is: " + list_labels[0] + ". In this case, it is important that the name you now make is different from this name."
        else:
            self.existing_labels_text = "The existing names are: " + ", ".join(list_labels) + ". In this case, it is important that the name you now make is different from these names."
        return self.existing_labels_text
    
    def description_FA(self, FA_component_dict): 
    
        top_features = FA_component_dict.get("top", [])
        top_values = FA_component_dict.get("values_top", [])
        bottom_features = FA_component_dict.get("bottom", [])
        bottom_values = FA_component_dict.get("values_bottom", [])


        text = ''
        # --- TOP FEATURES (positive loadings)
        if top_features:
            text = "The factor is "
            descriptions = [self.describe_level_FA(value) + f"the feature that {feature}" for feature, value in zip(top_features, top_values)]
            text += ", ".join(descriptions) + ". "
                
        # --- BOTTOM FEATURES (negative loadings)
        if bottom_features:
            text += "The factor is "
            descriptions = [self.describe_level_FA(value) + f"the feature that {feature}" for feature, value in zip(bottom_features, bottom_values)]
            text += ", ".join(descriptions) + ". "

        return text

class FALabel(FA):
    @property
    def tell_it_how_to_answer(self):
        return [f"{self.describe_base}/few_shot/few_shot_FA_label.xlsx"] 

    @property
    def tell_it_what_it_knows(self):
        return [""]
    
    def __init__(self):
        self.MH = ModelHandler()
        self.entity_id = st.session_state.entity_id
        super().__init__()

    def tell_it_who_it_is(self) -> List[Dict[str, str]]:
        """
        Constant introduction messages for the assistant.

        Returns:
        List of dicts with keys "role" and "content".
        """
        intro = [
            {
                "role": "system",
                "content": (
                    "You are a leading data scientist who specialises in explanatory data analysis. \n"
                    f"You have recently conducted factor analyses that allows you to summarise various entities, "
                    "identifying key underlying factors that explain the observed correlations among various features about those entities. \n"
                    "In each case, you have identified which factors are negatively and positively associated with specific features and how strong that association is. \n"
                    "Your task now is to name factors resulting from a factor analysis based on the analysis. \n"
                ),
            }, 
            {                
                "role": "assistant",
                "content": (    
                    "Understood. I will follow your instructions to name the factors based on their associations."
                ),
            }
        ]

        return intro

    def get_prompt_messages(self):

        prompt = ("Those were really good. Exactly what I am looking for. \n"
                  "I am now going to give you one last labelling task to complete. \n"
                  "Again, the name must strictly follow the form x vs y, where x is the opposite of the name y. "
                  "The name should not have a negative connotation. "
                  "A combination of two adjectives for both x and y are preferred but if you can think of a single word for each then use that.\n"
                  "Three word descriptions for both x and y are fine as well, especially if they capture the meaning well.\n"
                  "Longer names are not allowed.\n"
                  "Output the name (in x vs y format) only. \n"
        )
        if self.existing_labels_text != "": 
            prompt += f"{self.existing_labels_text}\n"

        prompt += f"Here is the factor description in terms of the features of a {self.entity_id}: ```{self.synthetic_text}```"
        
        return [{"role": "user", "content": prompt}]



    def setup_messages(self) -> List[Dict[str, str]]:
        """Builds and returns a list of chat messages for model input."""
        messages = self.tell_it_who_it_is()

        # --- Load few-shots examples  ---
        messages += [{
            "role": "user",
            "content":"You are now going to name the factors that come from some factor analyses.\n"
                "The name must strictly follow the format: 'negative vs positive' association.\n"
                f"Specifically, the name should be of the form x vs y, where x is one adjective that is "
                "negatively associated with the features and y is one adjective that is positively associated with the features.\n"
                "The adjective x should be the opposite of the name y.\n"
                "Neither of the adjectives should have a negative connotation.\n"
                "Output the name (in x vs y format) only.\n"
                "A combination of two adjectives for both x and y are preferred but if you can think of a single word for each then use that.\n"
                "Three word descriptions for both x and y are fine as well, especially if they capture the meaning well.\n"
                "Longer names are not allowed.\n"
                "I will provide you with factor descriptions and you will return the names in the format specified an nothing else. "}
            , 
            {"role": "assistant",
            "content": "Understood. Please provide the first factor description."
            }]

        try:
            example_paths = self.tell_it_how_to_answer
            messages += self.get_messages_from_excel(example_paths)
        except FileNotFoundError as e:
            # FIXME: When merging with new_training, add the other exception type
            print(f"Example paths file not found: {e}")

        # --- Add prompt messages ---
        messages += self.get_prompt_messages()

        return messages

class QandAWordalisation(FA):
    @property
    def tell_it_how_to_answer(self):
        return [f"{self.describe_base}/few_shot/few_shot_QandA.xlsx"] 

    @property
    def tell_it_what_it_knows(self):
        return [""]
    
    def __init__(self):
        self.MH = ModelHandler()
        self.entity_id = st.session_state.entity_id
        super().__init__()

    def tell_it_who_it_is(self) -> List[Dict[str, str]]:
        """
        Constant introduction messages for the assistant.

        Returns:
        List of dicts with keys "role" and "content".
        """
        intro = [
            {
                "role": "system",
                "content": (
                    "You are a leading data scientist who specialises in explanatory data analysis. \n"
                    f"You have recently conducted factor analyses that allows you to summarise various entities, "
                    "identifying key underlying factors that explain the observed correlations among various features about those entities. \n"
                    "In each case, you have identified which factors are negatively and positively associated with specific features and how strong that association is. \n"
                    "You have also given a name to the factors resulting from a factor analysis. \n"
                    "The names have the form x vs y, where x is the opposite of the name y. \n"
                    "Now, for each case, you are going to generate a short summary of factors based on the description provided.\n" 
                ),
            }, 
            {                
                "role": "assistant",
                "content": (    
                    "Understood. I will use the descriptions to generate short summaries."
                ),
            }
        ]

        return intro

    def get_prompt_messages(self):
        prompt = (
            "Those were really good. Exactly what I am looking for.\n"
            "Now, I am now going to give you one last task to complete. \n"
            f"I will give you, firstly, a name of a factor in the form {self.wholefactor}. \n" \
            f"Second, I will give you a description of {self.wholefactor} in terms of the features of a given {self.entity}.\n"
            f"Your task is to write a two sentence summary explaining how we should interpret {self.factor}.\n"
            f"The first sentence should explain what it means to be {self.factor}.\n"
            f"The second sentence should contrast it with being the opposite of {self.factor}.\n"
            "The summary should be easy to understand for someone not familiar with factor analysis and not mention any technical terms.\n"
            "It should be lively and engaging. Give life to the description.\n"
        )
        prompt = prompt + f"{self.synthetic_text}"
        return [{"role": "user", "content": prompt}]
    

    def tell_it_what_data_to_use(self, article,entity,factor,wholefactor,FA_component_dict):

        question = f"what does it mean when {article} {entity} is described as {factor}?"
        self.factor = factor
        self.wholefactor = wholefactor
        self.entity= entity
        self.synthetic_text = "The name of the factor is: " + wholefactor + ". \n" 
        self.synthetic_text += "It was based on the following description: "
        self.synthetic_text += self.description_FA(FA_component_dict) + "\n\n"
        self.synthetic_text += "Now tell me, " + question + "\n\n"

        return self.synthetic_text 

    def setup_messages(self) -> List[Dict[str, str]]:
        """Builds and returns a list of chat messages for model input."""
        messages = self.tell_it_who_it_is()

        # --- Load what it knows from the factor analysis ---
        try:
            tell_it_what_it_knows_paths = self.tell_it_what_it_knows
            messages += self.get_messages_from_excel(tell_it_what_it_knows_paths)
        except:
            # FIXME: When merging with new_training, add the other exception type
            print(f"Describe paths file not found: ")
            print(self.tell_it_what_it_knows)

        # --- Load few-shots examples  ---
        messages += [{
            "role": "user",
            "content": (
                    "Let's now try some examples of your task. I will now provide you with two things.\n"
                    "First, a name of a factor in the form x vs y, where adjective x is the opposite of the adjective y. \n" \
                    "Second, a description of the factor in terms of the features of a given entity.\n"
                    "Your task is to write a two sentence summary explaining how we should interpret either x or y.\n"
                    "Sometimes you will be asked to explain x, other times y.\n"
                    "The first sentence should explain what it means to possess the adjective in question.\n"
                    "The second sentence should contrast it with its opposite.\n"
                    "The summary should be easy to understand for someone not familiar with factor analysis and not mention any technical terms.\n"
                    "It should be lively and engaging. Give life to the description.\n"
                 )}, 
            {"role": "assistant",
            "content": "Understood! Please provide the first factor name and description."
            }]

        try:
            example_paths = self.tell_it_how_to_answer
            messages += self.get_messages_from_excel(example_paths)
        except FileNotFoundError as e:
            # FIXME: When merging with new_training, add the other exception type
            print(f"Example paths file not found: {e}")

        # --- Add prompt messages ---
        messages += self.get_prompt_messages()

        return messages

class QandAWordalisation_from_text(Wordalisation):
    @property
    def tell_it_how_to_answer(self):
        return [f"{self.describe_base}/few_shot/few_shot_QandA_from_text.xlsx"] 

    @property
    def tell_it_what_it_knows(self):
        return [""]
    
    def __init__(self):
        self.MH = ModelHandler()
        self.entity_id = st.session_state.entity_id
        super().__init__()

    def tell_it_who_it_is(self) -> List[Dict[str, str]]:
        """
        Constant introduction messages for the assistant.

        Returns:
        List of dicts with keys "role" and "content".
        """
        intro = [
            {
                "role": "system",
                "content": (
                    "You are a data analyst. \n"
                    "You did a factor analysis and now you will generate questions and answers pairs. \n"
                    "You have extra information provided, the questions and answers pairs are from this text"
                    "First, you will be provided with a set of examples to show you the output format."
                ),
            }, 
            {                
                "role": "assistant",
                "content": (    
                    "Understood. I will use the examples to guide me to generate the output format for the questions and answers pairs."
                ),
            }
        ]

        return intro

    def get_prompt_messages(self):
        prompt = (
            "You have an informative text."
            "You deduce question and answer pairs about that text."
            "The questions should be about the text, and the answers should explain them. "
            "The questions should be simple and the answers should be easy to understand."
            f"Now do the same thing with the following: ```{self.synthetic_text}```"
        )
        
        return [{"role": "user", "content": prompt}]
    

    def tell_it_what_data_to_use(self, FA_list_label):

        self.synthetic_text = FA_list_label
        return self.synthetic_text 

    def setup_messages(self) -> List[Dict[str, str]]:
        """Builds and returns a list of chat messages for model input."""
        messages = self.tell_it_who_it_is()

        # --- Load few-shots examples  ---
        messages += [{
            "role": "user",
            "content": (
                "You have an informative text."
                "You deduce question and answer pairs about that text."
                "The questions should be about the text, and the answers should explain them. "
                "The questions should be simple and the answers should be easy to understand."
                )}, 
            {"role": "assistant",
            "content": "Understood. Please provide the text."
            }]

        try:
            example_paths = self.tell_it_how_to_answer
            messages += self.get_messages_from_excel(example_paths)
        except FileNotFoundError as e:
            # FIXME: When merging with new_training, add the other exception type
            print(f"Example paths file not found: {e}")

        # --- Add prompt messages ---
        messages += self.get_prompt_messages()

        return messages

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

    def transform_msgs_for_gemini2(self, msgs):
        """
        Transform custom message structure into a tuple:
        (system_instruction, history_messages, user_message)
        where history_messages and user_message are compatible with Gemini API.

        Handles two input formats:
        1. list: A list of GPT-style message dicts ({'role': 'user'/'assistant'/'system', 'content': '...'}).
        2. dict: A custom dict shape (e.g., {'system_instruction': '...', 'history': [...], 'content': '...'}).
        """
        system_instruction = None
        gemini_history = []
        user_message = None

        # If msgs is a list of GPT-like messages
        if isinstance(msgs, list):
            # 1. First, process all messages to extract system instruction and build the history
            # Note: The *last* message in this list is assumed to be the new user query.
            for m in msgs:
                role = m.get("role")
                content = m.get("content", "")

                if role == "system":
                 # System instructions are typically handled separately
                    system_instruction = content
                elif role == "assistant":
                    # Maps 'assistant' to 'model' for Gemini
                    gemini_history.append({"role": "model", "parts": [{"text": content}]})
                elif role == "user":
                    # User messages go into history *if* they are not the last message
                    gemini_history.append({"role": "user", "parts": [{"text": content}]})
                # Ignore any other roles
        
            # 2. Extract the last user message as the new query and *remove* it from history
            if gemini_history and gemini_history[-1]["role"] == "user":
                # The user message content is the text part of the last message
                last_parts = gemini_history[-1]["parts"]
                if last_parts and isinstance(last_parts[0], dict) and "text" in last_parts[0]:
                    user_message = last_parts[0]["text"]
            
                # The fix: Remove the last user message from the history list
                gemini_history.pop()

        # Otherwise assume it's the custom dict shape
        elif isinstance(msgs, dict):
            system_instruction = msgs.get("system_instruction")

            # Process history messages from the 'history' key
            for msg in msgs.get("history", []):
                role = msg.get("role")
            
                # Ensure proper Gemini role naming
                if role == "assistant":
                    role = "model"
                elif role not in ("user", "model"):
                    continue # Skip non-user/model/assistant roles
            
                parts = msg.get("parts")
                # The custom dict shape assumes 'parts' can be a list of strings or a single string/object
                # A more robust fix would handle multimodal parts, but based on your original
                # implementation, we'll join/stringify parts into a single text.
                if parts is None:
                    text = ""
                elif isinstance(parts, (list, tuple)):
                    text = " ".join(str(p) for p in parts)
                else:
                    text = str(parts)

                # A message part in Gemini is an object, typically {"text": "..."}
                gemini_history.append({"role": role, "parts": [{"text": text}]})

            # Process the final user message from the 'content' key
            user_msg_obj = msgs.get("content")
            if user_msg_obj:
                parts = user_msg_obj.get("parts")
                if parts is None:
                    user_message = None
                elif isinstance(parts, (list, tuple)):
                    user_message = " ".join(str(p) for p in parts)
                else:
                    user_message = str(parts)
        # user_message will be None if user_msg_obj is None or if parts is None/empty

        else:
        # Handle unexpected type
            raise TypeError(f"Unexpected msgs type: {type(msgs)}. Expected list or dict.")
        
    
        return system_instruction, gemini_history, user_message


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
    
    