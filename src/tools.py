import openai
import base64
import os
import io
import time
from PIL import Image
from abc import ABCMeta, abstractmethod

# Créer une interaction pour un modèle de vision par ordinateur, prenant en compte une image en base64 et une requête textuelle
def create_vision_chat_completion(vision_model, base64_image, prompt):
    try:
        response = openai.ChatCompletion.create(
            model=vision_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                            },
                        },
                    ],
                }
            ],
            max_tokens=1000,
        )
        return response.choices[0].message.content
    except:
        return None

# Créer une image à partir d'un texte descriptif en utilisant le modèle DALL-E-3
def create_image(prompt):
    try:
        response = openai.Image.create(
            model="dall-e-3",
            prompt=prompt,
            response_format="b64_json"
        )
        return response.data[0]['b64_json']
    except:
        return None

# Convertir une image sur disque en chaîne base64
def image_to_base64(path):
    try:
        _, suffix = os.path.splitext(path)
        if suffix not in {'.jpg', '.jpeg', '.png', '.webp'}:
            img = Image.open(path)
            img_png = img.convert('RGB')
            img_png.tobytes()
            byte_buffer = io.BytesIO()
            img_png.save(byte_buffer, 'PNG')
            encoded_string = base64.b64encode(byte_buffer.getvalue()).decode('utf-8')
        else:
            with open(path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return encoded_string
    except:
        return None

# Convertir une chaîne base64 en bytes d'image
def base64_to_image_bytes(image_base64):
    try:
        return base64.b64decode(image_base64)
    except:
        return None

# Interroger un modèle de vision sur le contenu d'une image et afficher la réponse
def inquire_image(work_dir, vision_model, path, prompt):
    image_base64 = image_to_base64(f'{work_dir}/{path}')
    hypertext_to_display = None
    if image_base64 is None:
        return "Erreur: Erreur de générartion d'image", None
    else:
        response = create_vision_chat_completion(vision_model, image_base64, prompt)
        if response is None:
            return "Le modèle ne répond pas", None
        else:
            return response, hypertext_to_display

# Utiliser le modèle DALL-E pour générer une image et la sauvegarder
def dalle(unique_id, prompt):
    img_base64 = create_image(prompt)
    text_to_gpt = "L'image a été générée avec succès et affichée à l'utilisateur."

    if img_base64 is None:
        return "Erreur: le modèle ne répond pas", None

    img_bytes = base64_to_image_bytes(img_base64)
    if img_bytes is None:
        return "Erreur: Erreur de générartion d'image", None

    temp_path = f'cache/temp_{unique_id}'
    if not os.path.exists(temp_path):
        os.mkdir(temp_path)
    path = f'{temp_path}/{hash(time.time())}.png'

    with open(path, 'wb') as f:
        f.write(img_bytes)

    hypertext_to_display = f'<img src=\"file={path}\" width="50%" style=\'max-width:none; max-height:none\'>'
    return text_to_gpt, hypertext_to_display

# Classe abstraite pour les outils utilisant des modèles d'IA
class Tool(metaclass=ABCMeta):
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def support(self):
        pass

    @abstractmethod
    def get_tool_data(self):
        pass

# Outil pour interroger sur le contenu des images
class ImageInquireTool(Tool):
    def support(self):
        return self.config['model']['GPT-4V']['available']

    def get_tool_data(self):
        return {
            "tool_name": "inquire_image",
            "tool": inquire_image,
            "system_prompt": "Utilisez l'outil 'inquire_image' pour questionner un modèle IA à propos du contenu des images téléchargées par les utilisateurs. Évitez les phrases comme \"basé sur l'analyse\"; répondez plutôt comme si vous aviez vu l'image vous-même. Gardez à l'esprit que toutes les tâches liées aux images ne nécessitent pas de connaître le contenu de l'image, comme la conversion de formats ou l'extraction d'attributs de fichiers d'images, qui devraient utiliser l'outil `execute_code` à la place. Utilisez l'outil seulement lorsque la compréhension du contenu de l'image est nécessaire.",
            "tool_description": {
                "name": "inquire_image",
                "description": "Cette fonction vous permet d'interroger un modèle IA sur le contenu d'une image et de recevoir la réponse du modèle.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Chemin du fichier de l'image"
                        },
                        "prompt": {
                            "type": "string",
                            "description": "La question que vous souhaitez poser au modèle IA à propos de l'image"
                        }
                    },
                    "required": ["path", "prompt"]
                }
            },
            "additional_parameters": {
                "work_dir": lambda bot_backend: bot_backend.jupyter_work_dir,
                "vision_model": self.config['model']['GPT-4V']['model_name']
            }
        }

# Outil pour générer des images d'art avec DALL-E
class DALLETool(Tool):
    def support(self):
        return True

    def get_tool_data(self):
        return {
            "tool_name": "dalle",
            "tool": dalle,
            "system_prompt": "Si l'utilisateur vous demande de générer une image artistique, vous pouvez traduire les exigences de l'utilisateur en une description et l'envoyer à l'outil `dalle`. Notez que cet outil est spécifiquement conçu pour la création d'images artistiques. Pour les figures scientifiques, telles que les graphiques, veuillez utiliser l'outil d'exécution de code Python `execute_code` à la place.",
            "tool_description": {
                "name": "dalle",
                "description": "Cette fonction vous permet d'accéder au modèle DALL·E-3 d'OpenAI pour la génération d'images.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "Une description détaillée de l'image que vous souhaitez générer, doit être en anglais uniquement."
                        }
                    },
                    "required": ["prompt"]
                }
            },
            "additional_parameters": {
                "unique_id": lambda bot_backend: bot_backend.unique_id,
            }
        }

# Récupérer les outils disponibles selon la configuration
def get_available_tools(config):
    tools = [ImageInquireTool]

    available_tools = []
    for tool in tools:
        tool_instance = tool(config)
        if tool_instance.support():
            available_tools.append(tool_instance.get_tool_data())
    return available_tools
