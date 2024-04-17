from bot_backend import *
import base64
import time
import tiktoken
from notebook_serializer import add_code_cell_error_to_notebook, add_image_to_notebook, add_code_cell_output_to_notebook

# Message utilisé pour indiquer que la conversation a été tronquée pour tenir dans la fenêtre de tokens
SLICED_CONV_MESSAGE = "[Le reste de la conversation a été omis pour s'intégrer dans la fenêtre contextuelle.]"

def get_conversation_slice(conversation, model, encoding_for_which_model, min_output_tokens_count=500):
    """
    Extrait une portion de la conversation qui s'adapte à la limite de tokens du modèle utilisé.
    Cette fonction garde le premier message complet et autant de messages récents que possible.
    Paramètres :
    - conversation : Liste des messages échangés
    - model : Le modèle de GPT utilisé
    - encoding_for_which_model : Encodage spécifique au modèle
    - min_output_tokens_count : Nombre minimal de tokens réservés pour la réponse du modèle
    Retourne un tuple contenant la conversation tronquée, le nombre total de tokens, et un booléen indiquant si la troncature a eu lieu.
    """
    encoder = tiktoken.encoding_for_model(encoding_for_which_model)
    count_tokens = lambda txt: len(encoder.encode(txt))
    nb_tokens = count_tokens(conversation[0]['content'])
    sliced_conv = [conversation[0]]
    context_window_limit = int(config['model_context_window'][model])
    max_tokens = context_window_limit - count_tokens(SLICED_CONV_MESSAGE) - min_output_tokens_count
    sliced = False
    for message in conversation[-1:0:-1]:
        nb_tokens += count_tokens(message['content'])
        if nb_tokens > max_tokens:
            sliced_conv.insert(1, {'role': 'system', 'content': SLICED_CONV_MESSAGE})
            sliced = True
            break
        sliced_conv.insert(1, message)
    return sliced_conv, nb_tokens, sliced

def chat_completion(bot_backend: BotBackend):
    """
    Réalise une complétion de chat en utilisant le modèle spécifié dans bot_backend.
    Gère la troncature de la conversation pour s'adapter à la fenêtre de contexte du modèle.
    """
    model_choice = bot_backend.gpt_model_choice
    model_name = bot_backend.config['model'][model_choice]['model_name']
    kwargs_for_chat_completion = copy.deepcopy(bot_backend.kwargs_for_chat_completion)
    if bot_backend.config['API_TYPE'] == "azure":
        kwargs_for_chat_completion['messages'], nb_tokens, sliced = \
            get_conversation_slice(
                conversation=kwargs_for_chat_completion['messages'],
                model=model_name,
                encoding_for_which_model='gpt-3.5-turbo' if model_choice == 'GPT-3.5' else 'gpt-4'
            )
    else:
        kwargs_for_chat_completion['messages'], nb_tokens, sliced = \
            get_conversation_slice(
                conversation=kwargs_for_chat_completion['messages'],
                model=model_name,
                encoding_for_which_model=model_name
            )

    bot_backend.update_token_count(num_tokens=nb_tokens)
    bot_backend.update_sliced_state(sliced=sliced)

    assert config['model'][model_choice]['available'], f"{model_choice} n'est pas accessible avec votre clé API"
    assert model_name in config['model_context_window'], f"{model_name} manque d'informations sur la fenêtre de contexte. Veuillez vérifier le fichier config.json."

    response = openai.ChatCompletion.create(**kwargs_for_chat_completion)
    return response

def add_code_execution_result_to_bot_history(content_to_display, history, unique_id):
    """
    Ajoute les résultats d'exécution de code à l'historique du bot, incluant texte et images.
    Gère les erreurs et assure que toute sortie est correctement enregistrée et formatée.
    """
    images, text = [], []
    error_occurred = False

    for mark, out_str in content_to_display:
        if mark in ('stdout', 'execute_result_text', 'display_text'):
            text.append(out_str)
            add_code_cell_output_to_notebook(out_str)
        elif mark in ('execute_result_png', 'execute_result_jpeg', 'display_png', 'display_jpeg'):
            if 'png' in mark:
                images.append(('png', out_str))
                add_image_to_notebook(out_str, 'image/png')
            else:
                add_image_to_notebook(out_str, 'image/jpeg')
                images.append(('jpg', out_str))
        elif mark == 'error':
            text.append(delete_color_control_char(out_str))
            error_occurred = True
            add_code_cell_error_to_notebook(out_str)
    text = '\n'.join(text).strip('\n')
    if error_occurred:
        history.append([None, f'❌Erreur de sortie:\n```shell\n{text}\n```'])
    else:
        history.append([None, f'✔️Sortie du terminal:\n```shell\n{text}\n```'])

    for filetype, img in images:
        image_bytes = base64.b64decode(img)
        temp_path = f'cache/temp_{unique_id}'
        if not os.path.exists(temp_path):
            os.mkdir(temp_path)
        path = f'{temp_path}/{hash(time.time())}.{filetype}'
        with open(path, 'wb') as f:
            f.write(image_bytes)
        width, height = get_image_size(path)
        history.append(
            [
                None,
                f'<img src=\"file={path}\" style=\'{"" if width < 800 else "width: 800px;"} max-width:none; '
                f'max-height:none\'> '
            ]
        )

def add_function_response_to_bot_history(hypertext_to_display, history):
    """
    Ajoute la réponse d'une fonction sous forme d'hypertexte à l'historique du bot.
    Permet d'intégrer des réponses complexes et formatées provenant de fonctions externes.
    """
    if hypertext_to_display is not None:
        if history[-1][1]:
            history.append([None, hypertext_to_display])
        else:
            history[-1][1] = hypertext_to_display

def parse_json(function_args: str, finished: bool):
    """
    Analyse une chaîne JSON potentiellement non standard pour extraire le code à exécuter.
    Cette fonction est particulièrement utile pour traiter les chaînes JSON qui incluent des sauts de ligne incorrects ou des erreurs de formatage.
    """
    parser_log = {
        'met_begin_{': False,
        'begin_"code"': False,
        'end_"code"': False,
        'met_:': False,
        'met_end_}': False,
        'met_end_code_"': False,
        "code_begin_index": 0,
        "code_end_index": 0
    }
    try:
        for index, char in enumerate(function_args):
            if char == '{':
                parser_log['met_begin_{'] = True
            elif parser_log['met_begin_{'] and char == '"':
                if parser_log['met_:']:
                    if finished:
                        parser_log['code_begin_index'] = index + 1
                        break
                    else:
                        if index + 1 == len(function_args):
                            return None
                        else:
                            temp_code_str = function_args[index + 1:]
                            if '\n' in temp_code_str:
                                try:
                                    return json.loads(function_args + '"}')['code']
                                except json.JSONDecodeError:
                                    try:
                                        return json.loads(function_args + '}')['code']
                                    except json.JSONDecodeError:
                                        try:
                                            return json.loads(function_args)['code']
                                        except json.JSONDecodeError:
                                            if temp_code_str[-1] in ('"', '\n'):
                                                return None
                                            else:
                                                return temp_code_str.strip('\n')
                            else:
                                return json.loads(function_args + '"}')['code']
                elif parser_log['begin_"code"']:
                    parser_log['end_"code"'] = True
                else:
                    parser_log['begin_"code"'] = True
            elif parser_log['end_"code"'] and char == ':':
                parser_log['met_:'] = True
            else:
                continue
        if finished:
            for index, char in enumerate(function_args[::-1]):
                back_index = -1 - index
                if char == '}':
                    parser_log['met_end_}'] = True
                elif parser_log['met_end_}'] and char == '"':
                    parser_log['code_end_index'] = back_index - 1
                    break
                else:
                    continue
            code_str = function_args[parser_log['code_begin_index']: parser_log['code_end_index'] + 1]
            if '\n' in code_str:
                return code_str.strip('\n')
            else:
                return json.loads(function_args)['code']

    except Exception as e:
        return None

def get_image_size(image_path):
    """
    Obtient les dimensions d'une image à partir de son chemin.
    Retourne la largeur et la hauteur de l'image.
    """
    with Image.open(image_path) as img:
        width, height = img.size
    return width, height
