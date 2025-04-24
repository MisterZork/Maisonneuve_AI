import os
import tomllib

def lang():
    """
    Returns the current language setting.
    """
    return os.getenv("LANG", "en_US.UTF-8").split(".")[0]

def lang_str(tag, language=lang(), data_path="../data"):
    """
    Returns the string corresponding to that tag, depending on the language.
    :param tag: int: The tag corresponding to the string needed.
    :param language: str: The language for the code.
    :param data_path: str: The path to the data directory.
    :return: str: The language code.
    """
    with open(f"{data_path}/lang/{language}.toml", "rb") as f:
        lang_dict = tomllib.load(f)
    return lang_dict[tag] if tag in lang_dict else "N/A"


if __name__ == "__main__":
    print(lang())
    print(lang_str('1'))