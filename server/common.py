import glob
import json
import os
import yaml 
import functools

from flask import request, session

from models import UserStudy

def load_system_config(file_name="config.yaml"):
    location = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    with open(os.path.join(location, file_name), "r") as f:
        config = yaml.safe_load(f)

    # Fixup all the paths inside config to be absolute
    if not os.path.isabs(config["datasets_base_path"]):
        config["datasets_base_path"] = os.path.join(os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__))), config["datasets_base_path"])

    return config

yaml_config = load_system_config()
print(f"Yaml config={yaml_config}")

from urllib.parse import urlparse, urljoin

def gen_url_prefix():
    p = urlparse(request.url, ".")
    return f"{p.scheme}://{p.netloc}"
    #return f"http://{yaml_config['hostname']}:{yaml_config['port']}"

def load_languages(base_path):
    res = {}
    for lang in [x
                    for x in os.listdir(os.path.join(base_path, "static/languages"))
                    if os.path.isdir(os.path.join(base_path, "static/languages", x))
                ]:
        for x in glob.glob(os.path.join(base_path, f"static/languages/{lang}/*.json")):
            with open(x, "r", encoding="utf8") as f:
                if lang not in res:
                    res[lang] = dict()
                res[lang].update(json.loads(f.read()))
    return res

# Returns translator function for translating phrases to given language
def get_tr(languages, lang):
    def tr(phrase, alternative_phrase=None):
        if lang in languages and phrase in languages[lang]:
            return languages[lang][phrase]
        return alternative_phrase or phrase # Otherwise, if there is no translation, return alternative phrase (if specified) or original phrase.
    return tr

def multi_lang(func):
    @functools.wraps(func)
    def inner(*args, **kwargs):
        lang = request.args.get('lang')
        print(f"### Language = '{lang}'")
        if lang:
            session["lang"] = lang
        return func(*args, **kwargs)
    return inner

@functools.lru_cache(maxsize=None)
def load_user_study_config(user_study_id):
    user_study = UserStudy.query.filter(UserStudy.id == user_study_id).first()
    if not user_study:
        return None
    return json.loads(user_study.settings)

@functools.lru_cache(maxsize=None)
def load_user_study_config_by_guid(guid):
    user_study = UserStudy.query.filter(UserStudy.guid == guid).first()
    if not user_study:
        return None
    return json.loads(user_study.settings)