import gradio as gr
from response_parser import *

# Initialisation du dictionnaire d'état et du cache si nécessaire
def initialization(state_dict: Dict) -> None:
    if not os.path.exists('cache'):
        os.mkdir('cache')
    if state_dict["bot_backend"] is None:
        state_dict["bot_backend"] = BotBackend()
        if 'OPENAI_API_KEY' in os.environ:
            del os.environ['OPENAI_API_KEY']

# Récupération du backend de bot à partir du dictionnaire d'état
def get_bot_backend(state_dict: Dict) -> BotBackend:
    return state_dict["bot_backend"]

# Changement de modèle GPT utilisé selon le booléen `whether_switch`
def switch_to_gpt4(state_dict: Dict, whether_switch: bool) -> None:
    bot_backend = get_bot_backend(state_dict)
    if whether_switch:
        bot_backend.update_gpt_model_choice("GPT-4")
    else:
        bot_backend.update_gpt_model_choice("GPT-3.5")

# Ajout d'un message texte et mise à jour de l'historique
def add_text(state_dict: Dict, history: List, text: str) -> Tuple[List, Dict]:
    bot_backend = get_bot_backend(state_dict)
    bot_backend.add_text_message(user_text=text)

    history = history + [(text, None)]

    return history, gr.update(value="", interactive=False)

# Ajout d'un fichier et mise à jour de l'historique
def add_file(state_dict: Dict, history: List, files) -> List:
    bot_backend = get_bot_backend(state_dict)
    for file in files:
        path = file.name
        filename = os.path.basename(path)

        bot_msg = [f'📁[{filename}]', None]
        history.append(bot_msg)

        bot_backend.add_file_message(path=path, bot_msg=bot_msg)

        _, suffix = os.path.splitext(filename)
        if suffix in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}:
            copied_file_path = f'{bot_backend.jupyter_work_dir}/{filename}'
            width, height = get_image_size(copied_file_path)
            bot_msg[0] += \
                f'\n<img src=\"file={copied_file_path}\" style=\'{"" if width < 800 else "width: 800px;"} max-width' \
                f':none; max-height:none\'> '

    return history

# Annulation du dernier fichier téléchargé
def undo_upload_file(state_dict: Dict, history: List) -> Tuple[List, Dict]:
    bot_backend = get_bot_backend(state_dict)
    bot_msg = bot_backend.revoke_file()

    if bot_msg is None:
        return history, gr.Button.update(interactive=False)

    else:
        assert history[-1] == bot_msg
        del history[-1]
        if bot_backend.revocable_files:
            return history, gr.Button.update(interactive=True)
        else:
            return history, gr.Button.update(interactive=False)

# Actualisation de l'affichage des fichiers
def refresh_file_display(state_dict: Dict) -> List[str]:
    bot_backend = get_bot_backend(state_dict)
    work_dir = bot_backend.jupyter_work_dir
    filenames = os.listdir(work_dir)
    paths = []
    for filename in filenames:
        path = os.path.join(work_dir, filename)
        if not os.path.isdir(path):
            paths.append(path)
    return paths

# Actualisation du compteur de tokens
def refresh_token_count(state_dict: Dict):
    bot_backend = get_bot_backend(state_dict)
    model_choice = bot_backend.gpt_model_choice
    sliced = bot_backend.sliced
    token_count = bot_backend.context_window_tokens
    token_limit = config['model_context_window'][config['model'][model_choice]['model_name']]
    display_text = f"**Contexte de tokens:** {token_count}/{token_limit}"
    if sliced:
        display_text += '\\\nLimite de tokens dépassée, la conversion a été segmentée.'
    return gr.Markdown.update(value=display_text)

# Redémarrage de l'interface utilisateur
def restart_ui(history: List) -> Tuple[List, Dict, Dict, Dict, Dict, Dict, Dict]:
    history.clear()
    return (
        history,
        gr.Textbox.update(value="", interactive=False),
        gr.Button.update(interactive=False),
        gr.Button.update(interactive=False),
        gr.Button.update(interactive=False),
        gr.Button.update(interactive=False),
        gr.Button.update(visible=False)
    )

# Redémarrage du backend du bot
def restart_bot_backend(state_dict: Dict) -> None:
    bot_backend = get_bot_backend(state_dict)
    bot_backend.restart()

# Arrêt de la génération en cours
def stop_generating(state_dict: Dict) -> None:
    bot_backend = get_bot_backend(state_dict)
    if bot_backend.code_executing:
        bot_backend.send_interrupt_signal()
    else:
        bot_backend.update_stop_generating_state(stop_generating=True)

# Gestion des interactions et des réponses du bot
def bot(state_dict: Dict, history: List) -> List:
    bot_backend = get_bot_backend(state_dict)

    while bot_backend.finish_reason in ('new_input', 'function_call'):
        if history[-1][1]:
            history.append([None, ""])
        else:
            history[-1][1] = ""

        try:
            response = chat_completion(bot_backend=bot_backend)
            for chunk in response:
                if chunk['choices'] and chunk['choices'][0]['finish_reason'] == 'function_call':
                    if bot_backend.function_name in bot_backend.jupyter_kernel.available_functions:
                        yield history, gr.Button.update(value='⏹️ Interrupt execution'), gr.Button.update(visible=False)
                    else:
                        yield history, gr.Button.update(interactive=False), gr.Button.update(visible=False)

                if bot_backend.stop_generating:
                    response.close()
                    if bot_backend.content:
                        bot_backend.add_gpt_response_content_message()
                    if bot_backend.display_code_block:
                        bot_backend.update_display_code_block(
                            display_code_block="\n⚫Arrêté:\n```python\n{}\n```".format(bot_backend.code_str)
                        )
                        history = copy.deepcopy(bot_backend.bot_history)
                        history[-1][1] += bot_backend.display_code_block
                        bot_backend.add_function_call_response_message(function_response=None)

                    bot_backend.reset_gpt_response_log_values()
                    break

                history, whether_exit = parse_response(
                    chunk=chunk,
                    history=history,
                    bot_backend=bot_backend
                )

                yield (
                    history,
                    gr.Button.update(
                        interactive=False if bot_backend.stop_generating else True,
                        value='⏹️ Arrêter la génération'
                    ),
                    gr.Button.update(visible=False)
                )
                if whether_exit:
                    exit(-1)
        except openai.OpenAIError as openai_error:
            bot_backend.reset_gpt_response_log_values(exclude=['finish_reason'])
            yield history, gr.Button.update(interactive=False), gr.Button.update(visible=True)
            raise openai_error

    yield history, gr.Button.update(interactive=False, value='⏹️ Arrêter la génération'), gr.Button.update(visible=False)

if __name__ == '__main__':
    config = get_config()
    with gr.Blocks(theme=gr.themes.Base()) as block:
        """
        Référence : https://www.gradio.app/guides/creating-a-chatbot-fast
        """
        # Composants de l'interface utilisateur
        state = gr.State(value={"bot_backend": None})
        with gr.Tab("Chat"):
            chatbot = gr.Chatbot([], elem_id="chatbot", label="AgentLLM", height=750)
            with gr.Row():
                with gr.Column(scale=0.85):
                    text_box = gr.Textbox(
                        show_label=False,
                        placeholder="Entrez du texte et appuyez sur entrée, ou téléchargez un fichier",
                        container=False
                    )
                with gr.Column(scale=0.15, min_width=0):
                    file_upload_button = gr.UploadButton("📁", file_count='multiple', file_types=['file'])
            with gr.Row(equal_height=True):
                with gr.Column(scale=0.08, min_width=0):
                    check_box = gr.Checkbox(label="Utiliser GPT-4", interactive=config['model']['GPT-4']['available'])
                with gr.Column(scale=0.314, min_width=0):
                    model_token_limit = config['model_context_window'][config['model']['GPT-3.5']['model_name']]
                    token_count_display_text = f"**Contexte de tokens:** 0/{model_token_limit}"
                    token_monitor = gr.Markdown(value=token_count_display_text)
                with gr.Column(scale=0.15, min_width=0):
                    retry_button = gr.Button(value='🔂Erreur OpenAI, cliquez ici pour réessayer', visible=False)
                with gr.Column(scale=0.15, min_width=0):
                    stop_generation_button = gr.Button(value='⏹️ Arrêter la génération', interactive=False)
                with gr.Column(scale=0.15, min_width=0):
                    restart_button = gr.Button(value='🔄 Redémarrer')
                with gr.Column(scale=0.15, min_width=0):
                    undo_file_button = gr.Button(value="↩️Annuler le téléchargement du fichier", interactive=False)
        with gr.Tab("Fichiers"):
            file_output = gr.Files()

        # Liaison des fonctions aux composants
        txt_msg = text_box.submit(add_text, [state, chatbot, text_box], [chatbot, text_box], queue=False).then(
            lambda: gr.Button.update(interactive=False), None, [undo_file_button], queue=False
        ).then(
            bot, [state, chatbot], [chatbot, stop_generation_button, retry_button]
        )
        txt_msg.then(fn=refresh_file_display, inputs=[state], outputs=[file_output])
        txt_msg.then(lambda: gr.update(interactive=True), None, [text_box], queue=False)
        txt_msg.then(fn=refresh_token_count, inputs=[state], outputs=[token_monitor])

        retry_button.click(lambda: gr.Button.update(visible=False), None, [retry_button], queue=False).then(
            bot, [state, chatbot], [chatbot, stop_generation_button, retry_button]
        ).then(
            fn=refresh_file_display, inputs=[state], outputs=[file_output]
        ).then(
            lambda: gr.update(interactive=True), None, [text_box], queue=False
        ).then(
            fn=refresh_token_count, inputs=[state], outputs=[token_monitor]
        )

        check_box.change(fn=switch_to_gpt4, inputs=[state, check_box]).then(
            fn=refresh_token_count, inputs=[state], outputs=[token_monitor]
        )

        file_msg = file_upload_button.upload(
            add_file, [state, chatbot, file_upload_button], [chatbot], queue=False
        )
        file_msg.then(lambda: gr.Button.update(interactive=True), None, [undo_file_button], queue=False)
        file_msg.then(fn=refresh_file_display, inputs=[state], outputs=[file_output])

        undo_file_button.click(
            fn=undo_upload_file, inputs=[state, chatbot], outputs=[chatbot, undo_file_button]
        ).then(
            fn=refresh_file_display, inputs=[state], outputs=[file_output]
        )

        stop_generation_button.click(fn=stop_generating, inputs=[state], queue=False).then(
            fn=lambda: gr.Button.update(interactive=False), inputs=None, outputs=[stop_generation_button], queue=False
        )

        restart_button.click(
            fn=restart_ui, inputs=[chatbot],
            outputs=[
                chatbot, text_box, restart_button, file_upload_button, undo_file_button, stop_generation_button,
                retry_button
            ]
        ).then(
            fn=restart_bot_backend, inputs=[state], queue=False
        ).then(
            fn=refresh_file_display, inputs=[state], outputs=[file_output]
        ).then(
            fn=lambda: (gr.Textbox.update(interactive=True), gr.Button.update(interactive=True),
                        gr.Button.update(interactive=True)),
            inputs=None, outputs=[text_box, restart_button, file_upload_button], queue=False
        ).then(
            fn=refresh_token_count,
            inputs=[state], outputs=[token_monitor]
        )

        block.load(fn=initialization, inputs=[state])

    block.queue()
    block.launch(inbrowser=True)
