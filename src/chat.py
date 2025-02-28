import streamlit as st
import openai
from itertools import groupby
from types import GeneratorType
import pandas as pd
import json
import google.generativeai as genai


openai.api_type = "azure"


class Chat:

    def __init__(self):

        if st.session_state.selected_entity == None:
            self.indice = 0
        else:
            self.indice = st.session_state.df_filtered.index.tolist().index(
                st.session_state.selected_entity
            )
        
        if "messages_to_display" not in st.session_state:
            st.session_state.messages_to_display = []
            self.messages_to_display = st.session_state.messages_to_display
        else:
            self.messages_to_display = st.session_state.messages_to_display

        super().__init__()

    def test_indice(self):
        return self.indice


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


        converted_msgs = convert_messages_format(messages)

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
                avatar = "owl.jpg"
            else:
                try:
                    avatar = st.session_state.user_info["picture"]
                except:
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
        

    def instruction_messages(self):
        """
        Instruction for the agent.
        """
        first_messages = [
            {"role": "system", "content": "You are data analyst."},
            {
                "role": "user",
                "content": (
                    "After these messages you will be interacting with a user of the platform. "
                    f"The user has selected some data, and the conversation will be about them. "
                    "You will receive relevant information to answer a user's questions and then be asked to provide a response. "
                    "All user messages will be prefixed with 'User:' and enclosed with ```. "
                    "When responding to the user, speak directly to them. "
                    "Use the information provided before the query  to provide 2 sentence answers."
                    " Do not deviate from this information or provide additional information that is not in the text returned by the functions."
                ),
            },
        ]
        return first_messages

    

    def get_input(self):
        """
        Get input from streamlit."""

        if x := st.chat_input(
            placeholder=f"What else would you like to know about the data?"
        ):
            if len(x) > 500:
                st.error(
                    f"Your message is too long ({len(x)} characters). Please keep it under 500 characters."
                )

            self.handle_input(x)

def testfunc():
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display previous messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input
    if user_input := st.chat_input("Say something..."):
        st.chat_message("user").markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        msgs = {
                "system_instruction": "You are a data analyst and scientist",
                    "history": [
                        {"role": "user", "parts": "You give a description of the data. Don't invent the data, talk only about the data you have seen.\n"
                         "You can also ask questions about the data.\n"
                         "If the data are missing ask the user to input the data",},
                        {"role": "model", "parts": "Sure!"},
                    ],
                    "content": {"role": "user", "parts": user_input}
                }

        # Generate response
        model = genai.GenerativeModel(
                    model_name="gemini-1.5-flash",
                    system_instruction=msgs["system_instruction"],
                    generation_config=GenerationConfig(max_output_tokens=50),
                )
        chat = model.start_chat(history=msgs["history"])
        response = chat.send_message(content=msgs["content"])

        with st.chat_message("assistant"):
            assistant_reply = response.candidates[0].content.parts[0].text
            st.markdown(assistant_reply)

        st.session_state.messages.append({"role": "assistant", "content": assistant_reply})