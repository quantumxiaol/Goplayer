# 从config.yaml中读取配置,config.yaml和readConfig.py在同一目录下

import yaml
import os
import sys

# # 获取当前脚本所在的目录
# current_dir = os.path.dirname(os.path.abspath(__file__))

# # 获取上级目录
# parent_dir = os.path.dirname(current_dir)

# # 拼接 config.yaml 的完整路径
# config_path = os.path.join(parent_dir, "config.yaml")
config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
def get_openai_config():
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
        # 获取 OpenAI 配置
    openai_config = config.get("OpenAI", {})
    model = openai_config.get("model")
    api_key = openai_config.get("api_key")
    api_base = openai_config.get("api_base")
    return model, api_key, api_base