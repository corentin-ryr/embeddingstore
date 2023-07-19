from typing import Any
import yaml
from dataclasses import is_dataclass, fields


class YamlParser:

    @classmethod
    def load_to_dataclass(cls, dataclass_type: type, yaml_file_path: str) -> Any:
        with open(yaml_file_path, 'r') as f:
            yaml_config = yaml.safe_load(f)
            return cls.__dict_to_dataclass(dataclass_type, yaml_config)

    @staticmethod
    def __dict_to_dataclass(dataclass_type: type, data: dict):
        obj = dataclass_type()
        field_list = fields(dataclass_type)
        for field in field_list:
            if field.name not in data:
                continue
            if is_dataclass(field.type):
                setattr(obj, field.name, YamlParser.__dict_to_dataclass(field.type, data[field.name]))
            else:
                setattr(obj, field.name, data[field.name])
        return obj
