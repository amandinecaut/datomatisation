import streamlit as st
import openai
from itertools import groupby
from types import GeneratorType
import pandas as pd
import json
import google.generativeai as genai
from description import *

openai.api_type = "azure"


class Chat:

    def __init__(self, state="empty"):

        if st.session_state.selected_entity == None:
            self.indice = 0
        else:
            self.indice = st.session_state.df_filtered.index.tolist().index(
                st.session_state.selected_entity
            )

        st.session_state.chat_state = state
        
        if "messages_to_display" not in st.session_state:
            st.session_state.messages_to_display = []
            self.messages_to_display = st.session_state.messages_to_display
        
        else: 
            pass
        
        self.messages_to_display = st.session_state.messages_to_display
        self.state = st.session_state.chat_state


    def instruction_messages(self):
        """
        Sets up the instructions to the agent. Should be overridden by subclasses.
        """
        return []

    def add_message(self, content, role="assistant", user_only=True, visible=True):
        """
        Used by app.py to start off the conversation with plots and descriptions.
        """
        message = {"role": role, "content": content}
        self.messages_to_display.append(message)

    def handle_input(self, input):
        """
        The main function that calls the GPT-4 API and processes the response.
        """

        # Get the instruction messages.
        messages = self.instruction_messages()

        # Add a copy of the user messages. This is to give the assistant some context.
        messages = messages + self.messages_to_display.copy()

        # Get relevant information from the user input and then generate a response.
        # This is not added to messages_to_display as it is not a message from the assistant.
        get_relevant_info = self.get_relevant_info(input)

        # Now add the user input to the messages. Don't add system information and system messages to messages_to_display.
        self.messages_to_display.append({"role": "user", "content": input})

        messages.append(
            {
                "role": "user",
                "content": f"Here is the relevant information to answer the users query: {get_relevant_info}\n\n```User: {input}```",
            }
        )

        # Remove all items in messages where content is not a string
        messages = [
            message for message in messages if isinstance(message["content"], str)
        ]

        # Show the messages in an expander
        st.expander("Chat transcript", expanded=False).write(messages)


        converted_msgs = description.convert_messages_format(messages)

        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel(
            model_name=GEMINI_CHAT_MODEL,
            system_instruction=converted_msgs["system_instruction"],
        )
        chat = model.start_chat(history=converted_msgs["history"])
        response = chat.send_message(content=converted_msgs["content"])

        answer = response.text

        message = {"role": "assistant", "content": answer}

        # Add the returned value to the messages.
        self.messages_to_display.append(message)

    def display_content(self, content):
        """
        Displays the content of a message in streamlit. Handles plots, strings, and StreamingMessages.
        """
        if isinstance(content, str):
            st.write(content)

        # Visual
        elif isinstance(content, Visual):
            content.show()

        else:
            # So we do this in case
            try:
                content.show()
            except:
                try:
                    st.write(content.get_string())
                except:
                    raise ValueError(
                        f"Message content of type {type(content)} not supported."
                    )

    def display_messages(self):
        """
        Displays visible messages in streamlit. Messages are grouped by role.
        If message content is a Visual, it is displayed in a st.columns((1, 2, 1))[1].
        If the message is a list of strings/Visuals of length n, they are displayed in n columns.
        If a message is a generator, it is displayed with st.write_stream
        Special case: If there are N Visuals in one message, followed by N messages/StreamingMessages in the next, they are paired up into the same N columns.
        """
        # Group by role so user name and avatar is only displayed once

        # st.write(self.messages_to_display)

        for key, group in groupby(self.messages_to_display, lambda x: x["role"]):
            group = list(group)

            if key == "assistant":
                #avatar = st.image('owl.jpg')
                avatar = "👩‍🎤"
            else:
                avatar = None

            message = st.chat_message(name=key, avatar=avatar)
            with message:
                for message in group:
                    content = message["content"]
                    self.display_content(content)
    

    def save_state(self):
        """
        Saves the conversation to session state.
        """
        st.session_state.messages_to_display = self.messages_to_display
        st.session_state.chat_state = self.state

class EntityChat(Chat):
    def __init__(self, state="empty"):
        if st.session_state.selected_entity == None:
            self.indice = 0
        else:
            self.indice = st.session_state.df_filtered.index.tolist().index(
                st.session_state.selected_entity
            )
      
        super().__init__(state=state)

    def get_input(self):
        """
        Get input from streamlit."""

        if x := st.chat_input(
            placeholder=f"What else would you like to know about the entity {self.indice}?"
        ):
            if len(x) > 500:
                st.error(
                    f"Your message is too long ({len(x)} characters). Please keep it under 500 characters."
                )

            self.handle_input(x)


    def instruction_messages(self):
        """
        Instruction for the agent.
        """
        first_messages = [
            {"role": "system", "content": "You are a UK-based football scout."},
            {
                "role": "user",
                "content": (
                    "After these messages you will be interacting with a user of the data analysis platform. "
                    f"The user has selected the entity {self.indice}, and the conversation will be about them. "
                    "You will receive relevant information to answer a user's questions and then be asked to provide a response. "
                    "All user messages will be prefixed with 'User:' and enclosed with ```. "
                    "When responding to the user, speak directly to them. "
                    "Use the information provided before the query  to provide 2 sentence answers."
                    " Do not deviate from this information or provide additional information that is not in the text returned by the functions."
                ),
            },
        ]
        return first_messages

    def get_relevant_info(self, query):

        # If there is no query then use the last message from the user
        if query == "":
            query = self.visible_messages[-1]["content"]

        ret_val = "Here is a description of the entity in terms of data: \n\n"
        describe =  CreateDescription()
        summary = describe.stream_gpt(self.indice)

        
        ret_val += describe.synthesize_text()
        print(describe.synthesize_text())

        ret_val += f"\n\nIf none of this information is relevent to the users's query then use the information below to remind the user about the chat functionality: \n"
   

        return ret_val
