"""
File: config.py
Author: Jonathan Swinnen
Last Updated: 2023-09-11
Description: Contains functions to load/parse config files.
"""

import importlib
import yaml


def read(cfg_files):
    out = {}
    for cfg_file in cfg_files:
        out = merge(out, parse_yaml(cfg_file))
    return out

def parse_yaml(cfg_file):
    with open(cfg_file) as f:
        return yaml.load(f, Loader=yaml.SafeLoader)

def to_yaml(cfg_raw):
    yaml_string=yaml.dump(cfg_raw)
    return yaml_string

def load_objects(cfg_raw):
    if type(cfg_raw) == dict:
        if cfg_raw.get("__class_name__", None):
            return _obj_from_dict(cfg_raw)
        elif cfg_raw.get("__const_name__", None):
            return _const_from_dict(cfg_raw)
        else:
            parsed_hparams = dict()
            for key in cfg_raw:
                parsed_hparams[key] = load_objects(cfg_raw[key])
            return parsed_hparams
    if type(cfg_raw) == list:
        parsed_hparams = []
        for el in cfg_raw:
            parsed_hparams.append(load_objects(el))
        return parsed_hparams
    return cfg_raw 


def merge(*cfgs):
    out = {}
    for cfg in cfgs:
        out = _update_dict(out, cfg)
    return out


def remove_lists(cfg):
    if type(cfg) == list:
        return { i : remove_lists(cfg[i]) for i in range(len(cfg)) }
    if type(cfg) == dict:
        out = {}
        for key in cfg.keys():
            out[key] = remove_lists(cfg[key])
        return out
    return cfg
    

def _update_dict(original, update):
    if type(original) != type(update):
        return update
    if type(update) == list:
        if len(update) == 0: return update
        if update[0] == "__append__":
            return original + update[1:]
        return update
    if type(update) == dict:
        for key in update.keys():
            if not update[key] == "__delete__":
                if key in original:
                    original[key] = _update_dict(original[key], update[key])
                else:
                    original[key] = update[key]
            else:
                if key in original: del original[key]
        return original
    return update

        
def flatten(cfg, parent_key="", out={}):
    keys = None
    if type(cfg) == dict:
        keys = cfg.keys()
    elif type(cfg) == list:
        keys = range(len(cfg))
    for key in keys:
        new_parent_key = key
        if parent_key != "": new_parent_key = parent_key + "." + str(key)
        if type(cfg[key]) == dict or type(cfg[key]) == list:
            flat_entry = flatten(cfg[key], new_parent_key, out)
            out = {**out, **flat_entry}
        else:
            out[new_parent_key] = cfg[key]
    return out

def _obj_from_dict(d):
    class_path = d["__class_name__"].split(".")
    module_name = ".".join(class_path[:-1])
    class_name = class_path[-1]
    
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    params = load_objects(d.get("__params__",{}))
    obj = class_(**params)
    return obj

def _const_from_dict(d):
    const_path = d["__const_name__"].split(".")
    module_name = ".".join(const_path[:-1])
    const_name = const_path[-1]
    
    module = importlib.import_module(module_name)
    const = getattr(module, const_name)
    return const