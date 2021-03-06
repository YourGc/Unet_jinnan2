# coding:utf-8

import yaml


def load_cfg_from_file(cfg_file):
    with open(cfg_file, 'r', encoding='UTF-8') as f:
        yaml_cfg = yaml.load(f)
    f.close()

    return yaml_cfg
