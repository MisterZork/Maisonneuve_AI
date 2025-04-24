import tomllib
import os

#=============== Config functions ===============#
def load_config(path=os.path.dirname(__file__)):
    """
    Load the configuration from the config.toml file, inside the data directory.
    :return: config (dict): The configuration dictionary.
    """
    try:
        with open(f'{path}/data/config.toml', 'rb') as f:
            config = tomllib.load(f)
        return config
    except FileNotFoundError:
        print("Configuration file not found. Creating a new one.")
        create_config()
        print("Configuration file created. Please fill it with the required parameters.")
        return None

def collect_config(file_path):
    """
    This function collects the configuration from the config.toml file.
    :param file_path:
    :return:
    """
    pass

def create_config():
    """
    This function creates the config.toml file if it does not exist.
    :param
    """
    try:
        with open('../deprecated/data/config.toml', 'w') as f:
            f.write('[config]')
    except FileNotFoundError:
        pass

def create_dir(config):
    """
    This function creates the data directory if it does not exist.
    :param config: dict: The configuration dictionary.
    """
    if not os.path.exists('data'):
        os.makedirs('data')
        print("Directory 'data' created.")
    else:
        print("Directory 'data' already exists.")

if __name__ == '__main__':
    print(os.path.dirname(__file__))
    print(__name__)
    load_config()
