import json
import copy
import shutil
from jupyter_backend import *
from tools import *
from typing import *
from notebook_serializer import add_markdown_to_notebook, add_code_cell_to_notebook

# Configuration des fonctions utilisables via l'API
functions = [
    {
        "name": "execute_code",
        "description": "Cette fonction permet d'exécuter du code Python et de récupérer la sortie du terminal. Si le code "
                       "produit une sortie image, la fonction retournera le texte '[image]'. Le code est envoyé à un "
                       "noyau Jupyter pour exécution. Le noyau restera actif après l'exécution, conservant toutes les "
                       "variables en mémoire.",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Le texte du code"
                }
            },
            "required": ["code"],
        }
    },
]

# Message système initial expliquant le rôle de l'IA
system_msg = '''Vous êtes un interprète de code AI.
Votre objectif est d'aider les utilisateurs à accomplir diverses tâches en exécutant du code Python.

Vous devriez :
1. Comprendre avec précision et à la lettre les exigences des utilisateurs.
2. Donner une brève description de ce que vous prévoyez de faire & appeler la fonction fournie pour exécuter le code.
3. Fournir une analyse des résultats basée sur la sortie de l'exécution.
4. En cas d'erreur, essayer de la corriger.
5. Répondre dans la même langue que l'utilisateur.

Note : Si l'utilisateur télécharge un fichier, vous recevrez un message système "User uploaded a file: filename". Utilisez le nom du fichier comme chemin dans le code.'''

# Chargement de la configuration à partir d'un fichier JSON
with open('config.json') as f:
    config = json.load(f)

# Configuration de la clé API
if not config['API_KEY']:
    config['API_KEY'] = os.getenv('OPENAI_API_KEY')
    os.unsetenv('OPENAI_API_KEY')

def get_config():
    """ Récupère la configuration actuelle. """
    return config

def config_openai_api(api_type, api_base, api_version, api_key):
    """ Configure l'API d'OpenAI avec les paramètres spécifiés. """
    openai.api_type = api_type
    openai.api_base = api_base
    openai.api_version = api_version
    openai.api_key = api_key

class GPTResponseLog:
    """ Classe pour logger les réponses du modèle GPT utilisé par le backend. """
    def __init__(self):
        self.assistant_role_name = ''
        self.content = ''
        self.function_name = None
        self.function_args_str = ''
        self.code_str = ''
        self.display_code_block = ''
        self.finish_reason = 'stop'
        self.bot_history = None
        self.stop_generating = False
        self.code_executing = False
        self.interrupt_signal_sent = False

    def reset_gpt_response_log_values(self, exclude=None):
        """ Réinitialise les valeurs du log, sauf celles spécifiées. """
        if exclude is None:
            exclude = []

        attributes = {'assistant_role_name': '',
                      'content': '',
                      'function_name': None,
                      'function_args_str': '',
                      'code_str': '',
                      'display_code_block': '',
                      'finish_reason': 'stop',
                      'bot_history': None,
                      'stop_generating': False,
                      'code_executing': False,
                      'interrupt_signal_sent': False}

        for attr_name in exclude:
            del attributes[attr_name]
        for attr_name, value in attributes.items():
            setattr(self, attr_name, value)

    def set_assistant_role_name(self, assistant_role_name: str):
        """ Définit le nom de rôle de l'assistant. """
        self.assistant_role_name = assistant_role_name

    def add_content(self, content: str):
        """ Ajoute du contenu à la réponse actuelle. """
        self.content += content

    def set_function_name(self, function_name: str):
        """ Définit le nom de la fonction à appeler. """
        self.function_name = function_name

    def copy_current_bot_history(self, bot_history: List):
        """ Copie l'historique actuel du bot pour référence future. """
        self.bot_history = copy.deepcopy(bot_history)

    def add_function_args_str(self, function_args_str: str):
        """ Ajoute des arguments à la fonction appelée sous forme de chaîne de caractères. """
        self.function_args_str += function_args_str

    def update_code_str(self, code_str: str):
        """ Met à jour la chaîne de code à exécuter. """
        self.code_str = code_str

    def update_display_code_block(self, display_code_block):
        """ Met à jour le bloc de code à afficher. """
        self.display_code_block = display_code_block

    def update_finish_reason(self, finish_reason: str):
        """ Met à jour la raison de la fin de génération. """
        self.finish_reason = finish_reason

    def update_stop_generating_state(self, stop_generating: bool):
        """ Met à jour l'état d'arrêt de génération. """
        self.stop_generating = stop_generating

    def update_code_executing_state(self, code_executing: bool):
        """ Met à jour l'état d'exécution de code. """
        self.code_executing = code_executing

    def update_interrupt_signal_sent(self, interrupt_signal_sent: bool):
        """ Met à jour si un signal d'interruption a été envoyé. """
        self.interrupt_signal_sent = interrupt_signal_sent


class BotBackend(GPTResponseLog):
    """ Classe backend pour le bot utilisant GPT, gérant l'exécution et la logistique des outils. """
    def __init__(self):
        super().__init__()
        self.unique_id = hash(id(self))
        self.jupyter_work_dir = f'cache/work_dir_{self.unique_id}'
        self.tool_log = f'cache/tool_{self.unique_id}.log'
        self.jupyter_kernel = JupyterKernel(work_dir=self.jupyter_work_dir)
        self.gpt_model_choice = "GPT-3.5"
        self.revocable_files = []
        self.system_msg = system_msg
        self.functions = copy.deepcopy(functions)
        self._init_api_config()
        self._init_tools()
        self._init_conversation()
        self._init_kwargs_for_chat_completion()

    def _init_conversation(self):
        """ Initialise la conversation avec le message système initial. """
        first_system_msg = {'role': 'system', 'content': self.system_msg}
        self.context_window_tokens = 0  # Nombre de tokens effectivement envoyés à GPT
        self.sliced = False  # Indique si la conversion est tronquée
        if hasattr(self, 'conversation'):
            self.conversation.clear()
            self.conversation.append(first_system_msg)
        else:
            self.conversation: List[Dict] = [first_system_msg]

    def _init_api_config(self):
        """ Initialise la configuration de l'API en fonction du fichier de configuration. """
        self.config = get_config()
        api_type = self.config['API_TYPE']
        api_base = self.config['API_base']
        api_version = self.config['API_VERSION']
        api_key = config['API_KEY']
        config_openai_api(api_type, api_base, api_version, api_key)

    def _init_tools(self):
        """ Initialise les outils supplémentaires disponibles pour le backend. """
        self.additional_tools = {}

        tool_datas = get_available_tools(self.config)
        if tool_datas:
            self.system_msg += '\n\nOutils supplémentaires:'

        for tool_data in tool_datas:
            system_prompt = tool_data['system_prompt']
            tool_name = tool_data['tool_name']
            tool_description = tool_data['tool_description']

            self.system_msg += f'\n{tool_name}: {system_prompt}'

            self.functions.append(tool_description)
            self.additional_tools[tool_name] = {
                'tool': tool_data['tool'],
                'additional_parameters': copy.deepcopy(tool_data['additional_parameters'])
            }
            for parameter, value in self.additional_tools[tool_name]['additional_parameters'].items():
                if callable(value):
                    self.additional_tools[tool_name]['additional_parameters'][parameter] = value(self)

    def _init_kwargs_for_chat_completion(self):
        """ Initialise les arguments pour la complétion de chat avec OpenAI. """
        self.kwargs_for_chat_completion = {
            'stream': True,
            'messages': self.conversation,
            'functions': self.functions,
            'function_call': 'auto'
        }

        model_name = self.config['model'][self.gpt_model_choice]['model_name']

        if self.config['API_TYPE'] == 'azure':
            self.kwargs_for_chat_completion['engine'] = model_name
        else:
            self.kwargs_for_chat_completion['model'] = model_name

    def _backup_all_files_in_work_dir(self):
        """ Sauvegarde tous les fichiers dans le répertoire de travail. """
        count = 1
        backup_dir = f'cache/backup_{self.unique_id}'
        while os.path.exists(backup_dir):
            count += 1
            backup_dir = f'cache/backup_{self.unique_id}_{count}'
        shutil.copytree(src=self.jupyter_work_dir, dst=backup_dir)

    def _clear_all_files_in_work_dir(self, backup=True):
        """ Efface tous les fichiers du répertoire de travail, avec une option de sauvegarde. """
        if backup:
            self._backup_all_files_in_work_dir()
        for filename in os.listdir(self.jupyter_work_dir):
            path = os.path.join(self.jupyter_work_dir, filename)
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)

    def _save_tool_log(self, tool_response):
        """ Enregistre la réponse d'un outil dans le log du bot. """
        with open(self.tool_log, 'a', encoding='utf-8') as log_file:
            log_file.write(f'Previous conversion: {self.conversation}\n')
            log_file.write(f'Model choice: {self.gpt_model_choice}\n')
            log_file.write(f'Tool name: {self.function_name}\n')
            log_file.write(f'Parameters: {self.function_args_str}\n')
            log_file.write(f'Response: {tool_response}\n')
            log_file.write('----------\n\n')

    def add_gpt_response_content_message(self):
        """ Ajoute la réponse contentieuse du GPT à l'historique de la conversation. """
        self.conversation.append(
            {'role': self.assistant_role_name, 'content': self.content}
        )
        add_markdown_to_notebook(self.content, title="Assistant")

    def add_text_message(self, user_text):
        """ Ajoute un message texte de l'utilisateur à l'historique de la conversation. """
        self.conversation.append(
            {'role': 'user', 'content': user_text}
        )
        self.revocable_files.clear()
        self.update_finish_reason(finish_reason='new_input')
        add_markdown_to_notebook(user_text, title="User")

    def add_file_message(self, path, bot_msg):
        """ Ajoute un message de fichier téléchargé par l'utilisateur à l'historique. """
        filename = os.path.basename(path)
        work_dir = self.jupyter_work_dir

        shutil.copy(path, work_dir)

        gpt_msg = {'role': 'system', 'content': f'User uploaded a file: {filename}'}
        self.conversation.append(gpt_msg)
        self.revocable_files.append(
            {
                'bot_msg': bot_msg,
                'gpt_msg': gpt_msg,
                'path': os.path.join(work_dir, filename)
            }
        )

    def add_function_call_response_message(self, function_response: Union[str, None], save_tokens=True):
        """ Ajoute la réponse d'une fonction appelée à l'historique de la conversation. """
        if self.code_str is not None:
            add_code_cell_to_notebook(self.code_str)

        self.conversation.append(
            {
                "role": self.assistant_role_name,
                "name": self.function_name,
                "content": self.function_args_str
            }
        )
        if function_response is not None:
            if save_tokens and len(function_response) > 500:
                function_response = f'{function_response[:200]}\n[La sortie est trop volumineuse, une partie du milieu est omise]\n' \
                                    f'Partie finale de la sortie:\n{function_response[-200:]}'
            self.conversation.append(
                {
                    "role": "function",
                    "name": self.function_name,
                    "content": function_response,
                }
            )
        self._save_tool_log(tool_response=function_response)

    def append_system_msg(self, prompt):
        """ Ajoute un message système à l'historique de la conversation. """
        self.conversation.append(
            {'role': 'system', 'content': prompt}
        )

    def revoke_file(self):
        """ Annule le dernier fichier téléchargé et le supprime de l'historique et du serveur. """
        if self.revocable_files:
            file = self.revocable_files[-1]
            bot_msg = file['bot_msg']
            gpt_msg = file['gpt_msg']
            path = file['path']

            assert self.conversation[-1] is gpt_msg
            del self.conversation[-1]

            os.remove(path)

            del self.revocable_files[-1]

            return bot_msg
        else:
            return None

    def update_gpt_model_choice(self, model_choice):
        """ Met à jour le choix du modèle GPT utilisé. """
        self.gpt_model_choice = model_choice
        self._init_kwargs_for_chat_completion()

    def update_token_count(self, num_tokens):
        """ Met à jour le nombre de tokens utilisés dans la fenêtre de contexte. """
        self.__setattr__('context_window_tokens', num_tokens)

    def update_sliced_state(self, sliced):
        """ Met à jour l'état de troncature de la conversation. """
        self.__setattr__('sliced', sliced)

    def send_interrupt_signal(self):
        """ Envoie un signal d'interruption au noyau Jupyter. """
        self.jupyter_kernel.send_interrupt_signal()
        self.update_interrupt_signal_sent(interrupt_signal_sent=True)

    def restart(self):
        """ Redémarre le backend du bot, réinitialisant l'environnement et la conversation. """
        self.revocable_files.clear()
        self._init_conversation()
        self.reset_gpt_response_log_values()
        self.jupyter_kernel.restart_jupyter_kernel()
        self._clear_all_files_in_work_dir()
