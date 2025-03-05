import gradio as gr
from apis.llama_starter import multi_turn_with_history, single_turn


def launch_gradio():
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column(scale=1, min_width=150):
                gr.Markdown("# Modes")
                option_1 = gr.Button("Single-Turn Completion")
                option_2 = gr.Button("Multi-Turn Chat")
                option_3 = gr.Button("RAG Single-Turn")
                option_4 = gr.Button("RAG Multi-Turn")
            
            with gr.Column(scale=4) as main_frame:
                title = gr.Markdown("# 🛠 Single-Turn Completion")
                user_id = gr.Textbox(label="User ID", placeholder="Enter your User ID")
                chatbot = gr.Chatbot(render_markdown=True)
                user_input = gr.Textbox(label="Your Message", placeholder="Type here...")
                file_upload = gr.File(label="Upload File")
                send_button = gr.Button("Send")
                markdown_display = gr.Markdown()
                
                def multi_turn_with_history_v2(user_id, user_input, file):
                    return {"message": f"Response from v2 for {user_input}, File: {file.name if file else 'No File'}"}
                
                def multi_turn_with_history_v3(user_id, user_input, file):
                    return {"message": f"Response from v3 for {user_input}, File: {file.name if file else 'No File'}"}
                
                def on_send(user_id, user_input, chat_history, mode, file):
                    if not user_id or not user_input:
                        return chat_history + [(user_input, "Please enter User ID and message.")]
                    
                    if mode == "v1":
                        response = single_turn(user_id)
                    elif mode == "v2":
                        response = multi_turn_with_history(user_id, user_input)['message']
                    elif mode == "v3":
                        response = multi_turn_with_history_v3(user_id, user_input, file)['message']
                    elif mode == 'v4':
                        response = multi_turn_with_history_v2(user_id, user_input, file)['message']
                    
                    formatted_response = f"```markdown\n{response}\n```"
                    chat_history.append((user_input, response))
                    return chat_history, gr.update(value=formatted_response)
                
                chat_mode = gr.State("v1")
                chat_history = gr.State([])
                file_state = gr.State(None)
                
                send_button.click(on_send, [user_id, user_input, chat_history, chat_mode, file_upload], [chatbot, markdown_display])
            
            def switch_to_mode_1():
                return gr.update(value="# 🛠 Single-Turn Completion"), "v1", [], gr.update(value=""), [], gr.update(visible=False)
                
            def switch_to_mode_2():
                return gr.update(value="# 🛠 Multi-Turn Chat (History Aware)"), "v2", [], gr.update(value=""), [], gr.update(visible=False)
                
            def switch_to_mode_3():
                return gr.update(value="# 🛠 RAG Single-Turn"), "v3", [], gr.update(value=""), [], gr.update(visible=True)
            
            def switch_to_mode_4():
                return gr.update(value="# 🛠 RAG Multi-Turn"), "v4", [], gr.update(value=""), [], gr.update(visible=True)
                
            option_1.click(switch_to_mode_1, [], [title, chat_mode, chatbot, markdown_display, chat_history, file_upload])
            option_2.click(switch_to_mode_2, [], [title, chat_mode, chatbot, markdown_display, chat_history, file_upload])
            option_3.click(switch_to_mode_3, [], [title, chat_mode, chatbot, markdown_display, chat_history, file_upload])
            option_4.click(switch_to_mode_4, [], [title, chat_mode, chatbot, markdown_display, chat_history, file_upload])
    
    demo.launch()
