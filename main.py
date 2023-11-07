import logging
from functools import partial
import requests
from dotenv import load_dotenv
import os
import gradio as gr
import argparse
import datetime
import time
import torch
from assistant.library import Library
from assistant.gradio_css import code_highlight_css
from assistant.conversation import default_conversation
from assistant.serve_utils import (
    add_text,
    disable_btn,
    no_change_btn,
    downvote_last_response,
    enable_btn,
    flag_last_response,
    get_window_url_params,
    init,
    regenerate,
    upvote_last_response,
)
from assistant.model_utils import post_process_code
from assistant.chat_agent import ChatAgent
from langchain.schema.output_parser import StrOutputParser


load_dotenv()
logging.basicConfig(level=logging.INFO)

# library = Library()

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]


def load_demo(url_params, request: gr.Request):
    dropdown_update = gr.Dropdown(visible=True)
    state = default_conversation.copy()

    return (
        state,
        dropdown_update,
        gr.Chatbot(visible=True),
        gr.Textbox(visible=True),
        gr.Button(visible=True),
        gr.Row(visible=True),
        gr.Accordion(visible=True),
    )


def clear_history(request: gr.Request):
    state = default_conversation.copy()

    return (state, state.to_gradio_chatbot(), "", None, None) + (disable_btn,) * 5


def set_api_key(api_key):
    openai.api_key = api_key
    os.environ["OPENAI_API_KEY"] = api_key
    return "API Key set successfully."


def http_bot(
    state,
    max_output_tokens,
    temperature,
    top_k,
    top_p,
    num_beams,
    no_repeat_ngram_size,
    length_penalty,
    do_sample,
    request: gr.Request,
):
    if state.skip_next:
        # This generate call is skipped due to invalid inputs
        yield (state, state.to_gradio_chatbot()) + (no_change_btn,) * 5
        return

    prompt = state.get_prompt()

    data = {
        "text_input": prompt,
        "generation_config": {
            "top_k": int(top_k),
            "top_p": float(top_p),
            "num_beams": int(num_beams),
            "no_repeat_ngram_size": int(no_repeat_ngram_size),
            "length_penalty": float(length_penalty),
            "do_sample": bool(do_sample),
            "temperature": float(temperature),
            "max_new_tokens": min(int(max_output_tokens), 1536),
        },
    }

    state.messages[-1][-1] = "▌"
    yield (state, state.to_gradio_chatbot()) + (disable_btn,) * 5

    try:
        import pdb

        pdb.set_trace()
        for chunk in agent.predict(data):
            if chunk:
                if chunk[1]:
                    output = chunk[0].strip()
                    output = post_process_code(output)
                    state.messages[-1][-1] = output + "▌"
                    yield (state, state.to_gradio_chatbot()) + (disable_btn,) * 5
                else:
                    output = chunk[0].strip()
                    state.messages[-1][-1] = output
                    yield (state, state.to_gradio_chatbot()) + (
                        disable_btn,
                        disable_btn,
                        disable_btn,
                        enable_btn,
                        enable_btn,
                    )
                    return
                time.sleep(0.03)

    except requests.exceptions.RequestException as e:
        state.messages[-1][
            -1
        ] = "**NETWORK ERROR DUE TO HIGH TRAFFIC. PLEASE REGENERATE OR REFRESH THIS PAGE.**"
        yield (state, state.to_gradio_chatbot()) + (
            disable_btn,
            disable_btn,
            disable_btn,
            enable_btn,
            enable_btn,
        )
        return

    state.messages[-1][-1] = state.messages[-1][-1][:-1]
    yield (state, state.to_gradio_chatbot()) + (enable_btn,) * 5


def add_text_http_bot(
    state,
    text,
    max_output_tokens,
    temperature,
    top_k,
    top_p,
    num_beams,
    no_repeat_ngram_size,
    length_penalty,
    do_sample,
    request: gr.Request,
):
    if len(text) <= 0:
        state.skip_next = True
        return (state, state.to_gradio_chatbot(), "", None, None) + (no_change_btn,) * 5

    state.append_message(state.roles[0], text)
    state.append_message(state.roles[1], None)
    state.skip_next = False
    logging.info(state)
    yield (state, state.to_gradio_chatbot(), "", None, None) + (disable_btn,) * 5

    if state.skip_next:
        # This generate call is skipped due to invalid inputs
        yield (state, state.to_gradio_chatbot(), "", None, None) + (no_change_btn,) * 5
        return

    prompt = state.get_prompt()

    data = {
        "text_input": prompt,
        "generation_config": {
            "top_k": int(top_k),
            "top_p": float(top_p),
            "num_beams": int(num_beams),
            "no_repeat_ngram_size": int(no_repeat_ngram_size),
            "length_penalty": float(length_penalty),
            "do_sample": bool(do_sample),
            "temperature": float(temperature),
            "max_new_tokens": min(int(max_output_tokens), 1536),
        },
    }

    state.messages[-1][-1] = "▌"
    yield (state, state.to_gradio_chatbot(), "", None, None) + (disable_btn,) * 5

    # try:

    #     agent_ret = agent.predict(data)

    #     logging.info("Formatting agent response.")

    #     for chunk in agent_ret:
    #         output = chunk[0].strip()
    #         state.messages[-1][-1] = output + "▌"

    #         yield (state, state.to_gradio_chatbot(), "", None, None) + (disable_btn,) * 5
    #         return

    try:
        # # import pdb; pdb.set_trace()
        # for chunk in agent.predict(data):
        #     if chunk:
        #         output = chunk[0].strip()
        #         output = post_process_code(output)
        #         print(output)
        #         state.messages[-1][-1] = output + "▌"
        #         print(output + "▌")
        #         yield (state, state.to_gradio_chatbot(), "", None, None) + (
        #             disable_btn,
        #         ) * 5
        #     else:
        #         output = chunk[0].strip()
        #         state.messages[-1][-1] = output
        #         yield (state, state.to_gradio_chatbot(), "", None, None) + (
        #             disable_btn,
        #             disable_btn,
        #             disable_btn,
        #             enable_btn,
        #             enable_btn,
        #         )
        #         return
        #     time.sleep(0.03)
        # import pdb; pdb.set_trace()
        agent_return = agent.predict(data)

        for ret in agent_return:
            output = ret[0].strip()
            output = post_process_code(output)
            print(output)
            state.messages[-1][-1] = output + "▌"
            print(output + "▌")
            yield (state, state.to_gradio_chatbot(), "", None, None) + (disable_btn,) * 5

    except requests.exceptions.RequestException as e:
        state.messages[-1][
            -1
        ] = "**NETWORK ERROR DUE TO HIGH TRAFFIC. PLEASE REGENERATE OR REFRESH THIS PAGE.**"
        yield (state, state.to_gradio_chatbot(), "", None, None) + (
            disable_btn,
            disable_btn,
            disable_btn,
            enable_btn,
            enable_btn,
        )
        return
    state.messages[-1][-1] = state.messages[-1][-1]

    yield (state, state.to_gradio_chatbot(), "", None, None) + (enable_btn,) * 5


def regenerate_http_bot(
    state,
    max_output_tokens,
    temperature,
    top_k,
    top_p,
    num_beams,
    no_repeat_ngram_size,
    length_penalty,
    do_sample,
    request: gr.Request,
):
    state.messages[-1][-1] = None
    state.skip_next = False
    yield (state, state.to_gradio_chatbot(), "", None, None) + (disable_btn,) * 5

    prompt = state.get_prompt()

    data = {
        "text_input": prompt,
        "generation_config": {
            "top_k": int(top_k),
            "top_p": float(top_p),
            "num_beams": int(num_beams),
            "no_repeat_ngram_size": int(no_repeat_ngram_size),
            "length_penalty": float(length_penalty),
            "do_sample": bool(do_sample),
            "temperature": float(temperature),
            "max_new_tokens": min(int(max_output_tokens), 1536),
        },
    }

    state.messages[-1][-1] = "▌"
    yield (state, state.to_gradio_chatbot(), "", None, None) + (disable_btn,) * 5

    try:
        for chunk in agent.predict(data):
            if chunk:
                if chunk[1]:
                    output = chunk[0].strip()
                    output = post_process_code(output)
                    state.messages[-1][-1] = output + "▌"
                    yield (state, state.to_gradio_chatbot(), "", None, None) + (
                        disable_btn,
                    ) * 5
                else:
                    output = chunk[0].strip()
                    state.messages[-1][-1] = output
                    yield (state, state.to_gradio_chatbot(), "", None, None) + (
                        disable_btn,
                        disable_btn,
                        disable_btn,
                        enable_btn,
                        enable_btn,
                    )
                    return
                time.sleep(0.03)

    except requests.exceptions.RequestException as e:
        state.messages[-1][
            -1
        ] = "**NETWORK ERROR DUE TO HIGH TRAFFIC. PLEASE REGENERATE OR REFRESH THIS PAGE.**"
        yield (state, state.to_gradio_chatbot(), "", None, None) + (
            disable_btn,
            disable_btn,
            disable_btn,
            enable_btn,
            enable_btn,
        )
        return

    state.messages[-1][-1] = state.messages[-1][-1][:-1]
    yield (state, state.to_gradio_chatbot(), "", None, None) + (enable_btn,) * 5


css = (
    code_highlight_css
    + """
pre {
    white-space: pre-wrap;       /* Since CSS 2.1 */
    white-space: -moz-pre-wrap;  /* Mozilla, since 1999 */
    white-space: -pre-wrap;      /* Opera 4-6 */
    white-space: -o-pre-wrap;    /* Opera 7 */
    word-wrap: break-word;       /* Internet Explorer 5.5+ */
}
"""
)


with gr.Blocks(title="snkl assistant", css=css) as demo:
    state = gr.State()

    with gr.Row():
        with gr.Column(scale=3):
            with gr.Accordion("Parameters", open=True, visible=False) as parameter_row:
                max_output_tokens = gr.Slider(
                    minimum=0,
                    maximum=1024,
                    value=512,
                    step=64,
                    interactive=True,
                    label="Max output tokens",
                )
                temperature = gr.Slider(
                    minimum=0,
                    maximum=1,
                    value=1,
                    step=0.1,
                    interactive=True,
                    label="Temperature",
                )
                top_k = gr.Slider(
                    minimum=1,
                    maximum=5,
                    value=3,
                    step=1,
                    interactive=True,
                    label="Top K",
                )
                top_p = gr.Slider(
                    minimum=0,
                    maximum=1,
                    value=0.9,
                    step=0.1,
                    interactive=True,
                    label="Top p",
                )
                length_penalty = gr.Slider(
                    minimum=1,
                    maximum=5,
                    value=1,
                    step=0.1,
                    interactive=True,
                    label="length_penalty",
                )
                num_beams = gr.Slider(
                    minimum=1,
                    maximum=5,
                    value=1,
                    step=1,
                    interactive=True,
                    label="Beam Size",
                )
                no_repeat_ngram_size = gr.Slider(
                    minimum=1,
                    maximum=5,
                    value=2,
                    step=1,
                    interactive=True,
                    label="no_repeat_ngram_size",
                )
                do_sample = gr.Checkbox(interactive=True, value=True, label="do_sample")

                model_selection = gr.Dropdown(
                    choices=["gpt-4", "gpt-3.5-turbo"],
                    label="Select a GPT Model",
                    value="gpt-3.5-turbo",
                )

        with gr.Column(scale=6):
            chatbot = gr.Chatbot(elem_id="chatbot", visible=True, height=500)
            with gr.Row():
                with gr.Column(scale=8):
                    textbox = gr.Textbox(
                        show_label=True,
                        placeholder="Enter text and press ENTER",
                        visible=False,
                    )
                with gr.Column(scale=1, min_width=60):
                    submit_btn = gr.Button(value="Submit", visible=False)
            with gr.Row(visible=False) as button_row:
                upvote_btn = gr.Button(value="👍  Upvote", interactive=True)
                downvote_btn = gr.Button(value="👎  Downvote", interactive=True)
                flag_btn = gr.Button(value="⚠️  Flag", interactive=True)
                regenerate_btn = gr.Button(value="🔄  Regenerate", interactive=True)
                clear_btn = gr.Button(value="🗑️  Clear history", interactive=True)

    url_params = gr.JSON(visible=True)

    btn_list = [upvote_btn, downvote_btn, flag_btn, regenerate_btn, clear_btn]

    parameter_list = [
        max_output_tokens,
        temperature,
        top_k,
        top_p,
        num_beams,
        no_repeat_ngram_size,
        length_penalty,
        do_sample,
    ]

    upvote_btn.click(
        upvote_last_response, [state], [textbox, upvote_btn, downvote_btn, flag_btn]
    )

    downvote_btn.click(
        downvote_last_response,
        [state],
        [textbox, upvote_btn, downvote_btn, flag_btn],
    )

    flag_btn.click(
        flag_last_response, [state], [textbox, upvote_btn, downvote_btn, flag_btn]
    )

    regenerate_btn.click(
        regenerate_http_bot,
        [state] + parameter_list,
        [state, chatbot, textbox] + btn_list,
    )

    clear_btn.click(
        clear_history,
        None,
        [state, chatbot, textbox] + btn_list,
    )

    textbox.submit(
        add_text_http_bot,
        [state, textbox] + parameter_list,
        [state, chatbot, textbox] + btn_list,
    )

    submit_btn.click(
        add_text_http_bot,
        [state, textbox] + parameter_list,
        [state, chatbot, textbox] + btn_list,
    )

    demo.load(
        load_demo,
        [url_params],
        [state, chatbot, textbox, submit_btn, button_row, parameter_row],
        #_js=get_window_url_params,
    )


if __name__ == "__main__":
    io = init()
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = cur_dir[:-9] + "log"

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--debug", action="store_true", help="using debug mode")
    parser.add_argument("--port", type=int)
    parser.add_argument("--concurrency-count", type=int, default=100)
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    agent = ChatAgent(
        log_dir=log_dir,
        device=device,
        io=io,
    )

    demo.queue( status_update_rate=10, api_open=False
    )
    demo.launch(
        height=500,
        server_name=args.host,
        debug=args.debug,
        server_port=args.port,
        share=False,
    )
