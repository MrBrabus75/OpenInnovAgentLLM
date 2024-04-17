# AgentLLM
Une implémentation locale du Code Interpreter de ChatGPT d'OpenAI (Analyse de Données Avancée).

## Introduction

L'Interpréteur de Code d'OpenAI (récemment renommé en Analyse de Données Avancée) pour ChatGPT est une fonctionnalité révolutionnaire qui permet l'exécution de code Python à l'intérieur du modèle IA. Toutefois, il exécute le code dans un bac à sable en ligne et présente certaines limitations. Dans ce projet, nous présentons AgentLLM – qui permet l'exécution de code sur votre appareil local, offrant plus de flexibilité, de sécurité et de commodité.

## Principaux Avantages

- **Environnement Personnalisé** : Exécutez du code dans un environnement personnalisé de votre choix, en vous assurant d'avoir les bons paquets et configurations.

- **Expérience Transparente** : Dites adieu aux restrictions de taille de fichier et aux problèmes internet lors du téléchargement. Avec l'Interpréteur de Code Local, vous contrôlez entièrement.

- **Disponibilité de GPT-3.5** : Alors que l'Interpréteur de Code officiel est uniquement disponible pour le modèle GPT-4, l'Interpréteur de Code Local offre la flexibilité de basculer entre les modèles GPT-3.5 et GPT-4.

- **Sécurité des Données Améliorée** : Gardez vos données plus sécurisées en exécutant le code localement, minimisant le transfert de données sur internet.

- **Support de Jupyter** : Vous pouvez sauvegarder tout le code et l'historique des conversations dans un cahier Jupyter pour utilisation future.

## Note
Exécuter du code généré par IA sans révision humaine sur votre propre appareil n'est pas sûr. Vous êtes responsable de prendre des mesures pour protéger la sécurité de votre appareil et de vos données (telles que l'utilisation d'une machine virtuelle) avant de lancer ce programme. Toutes les conséquences causées par l'utilisation de ce programme seront à votre charge.

## Utilisation

### Installation

1. Clonez ce dépôt sur votre appareil local
   ```shell
   git clone https://github.com/MrBrabus75/OpenInnovAgentLLM.git
   cd AgentLLM
   ```

2. Installez les dépendances nécessaires. Le programme a été testé sur Windows 10 et CentOS Linux 7.8, avec Python 3.9.16. Les paquets requis incluent :
   ```text
   Jupyter Notebook    6.5.4
   gradio              3.39.0
   openai              0.27.8
   ansi2html           1.8.0
   tiktoken            0.3.3
   Pillow              9.4.0
   ```
   D'autres systèmes ou versions de paquets peuvent également fonctionner. Veuillez noter que vous ne devriez pas mettre à jour le paquet openai à la dernière version 1.x, car il a été réécrit et n'est pas compatible avec les versions antérieures.
Vous pouvez utiliser la commande suivante pour installer directement les paquets requis :
   ```shell
   pip install -r requirements.txt
   ```
   Pour les nouveaux utilisateurs de Python, nous proposons une commande pratique qui installe des paquets supplémentaires couramment utilisés pour le traitement et l'analyse des données :
   ```shell
   pip install -r requirements_full.txt
   ```
### Configuration

1. Créez un fichier `config.json` dans le répertoire src, en suivant les exemples fournis dans le répertoire `config_example`.

2. Configurez votre clé API dans le fichier `config.json`

Veuillez Noter :
1. **Réglez Correctement le `model_name`**
    Ce programme repose sur la capacité d'appel de fonction des versions `0613` ou plus récentes des modèles :
    - `gpt-3.5-turbo-0613` (et sa version 16K)
    - `gpt-3.5-turbo-1106`
    - `gpt-3.5-turbo-0125`
    - `gpt-4-0613` (et sa version 32K)
    - `gpt-4-1106-preview` 
    - `gpt-4-0125-preview`

    Les versions antérieures des modèles ne fonctionneront pas. Notez que `gpt-4-vision-preview` manque de support pour l'appel de fonction, donc, il ne devrait pas être réglé comme modèle `GPT-4`. 

    Pour les utilisateurs du service Azure OpenAI:
    - Réglez le `model_name` comme votre nom de déploiement.
    - Confirmez que le modèle déployé correspond à la version `0613` ou plus récente.

2. **Paramètres de Version de l'API**
    Si vous utilisez le service Azure OpenAI, réglez le `API_VERSION` à `2024-03-01-preview` dans le fichier `config.json`. Notez que les versions de l'API antérieures à `2023-07-01-preview` ne supportent pas les appels de fonction nécessaires pour ce programme et `2024-03-01-preview` est recommandée car les versions antérieures seront obsolètes dans un avenir proche.

3. **Paramètres du Modèle Vision**
    Bien que `gpt-4-vision-preview` ne supporte actuellement pas l'appel de fonction, nous avons implémenté l'entrée visuelle en utilisant une approche non de bout en bout. Pour activer l'entrée visuelle, réglez `gpt-4-vision-preview` comme modèle `GPT-4V` et réglez `available` à `true`. À l'inverse, réglez `available` à `false` pour désactiver l'entrée visuelle lorsque c'est inutile, ce qui éliminera les invites système liées à la vision et réduira vos coûts API.

4. **Paramètres de Fenêtre de Contexte du Modèle**
    Le champ `model_context_window` enregistre la fenêtre de contexte pour chaque modèle, que le programme utilise pour découper les conversations lorsqu'elles dépassent la capacité de fenêtre de contexte du modèle. 
    Les utilisateurs du service Azure OpenAI devraient manuellement insérer les informations de fenêtre de contexte en utilisant le nom de déploiement du modèle dans le format suivant :
    ```json
    "<YOUR-DEPLOYMENT-NAME>": <contex_window (integer)>
    ```
   
    De plus, lorsque OpenAI introduit de nouveaux modèles, vous pouvez manuellement ajouter les informations de fenêtre de contexte du nouveau modèle en utilisant le même format. (Nous garderons ce fichier à jour, mais il pourrait y avoir des retards)

5. **Gestion Alternative de la Clé API**
    Si vous préférez ne pas stocker votre clé API dans le fichier `config.json`, vous pouvez opter pour une approche alternative :
    - Laissez le champ `API_KEY` dans `config.json` comme une chaîne vide :
        ```json
        "API_KEY": ""
        ```
    - Réglez la variable d'environnement `OPENAI_API_KEY` avec votre clé API avant de lancer le programme :
        - Sur Windows :
        ```shell
        set OPENAI_API_KEY=<VOTRE-CLÉ-API>
        ```
        - Sur Linux :
        ```shell
        export OPENAI_API_KEY=<VOTRE-CLÉ-API>
        ```

## Pour Commencer

1. Naviguez vers le répertoire `src`.
   ```shell
   cd src
   ```

2. Exécutez la commande:
   ```shell
   python web_ui.py
   ```

3. Accédez au lien généré dans votre navigateur pour commencer à utiliser AgentLLM.

4. Utilisez l'option `-n` ou `--notebook` pour sauvegarder la conversation dans un cahier Jupyter.
   Par défaut, le cahier est sauvegardé dans le répertoire de travail, mais vous pouvez ajouter un chemin pour le sauvegarder ailleurs.
   ```shell
   python web_ui.py -n <path_to_notebook>
   ```
