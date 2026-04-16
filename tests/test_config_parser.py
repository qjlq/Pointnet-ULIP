import sys
sys.path.insert(0, '.')
from project_utils.config_parser import ConfigParser

def test_config_parser_loads_yaml():
    parser = ConfigParser()
    config = parser.load('config/fusion_config.yaml')
    assert 'data' in config
    assert 'models' in config

def test_config_val_parameters():
    parser = ConfigParser()
    config = parser.load('config/fusion_config.yaml')
    assert 'val_ratio' in config['extraction']
    assert config['extraction']['val_ratio'] == 0.2
    assert isinstance(config['extraction']['val_ratio'], (float, int))
    assert 'split_seed' in config['extraction']
    assert config['extraction']['split_seed'] == 42
    assert isinstance(config['extraction']['split_seed'], int)

def test_scanobjectnn_config_val_parameters():
    parser = ConfigParser()
    config = parser.load('config/scanobjectnn_config.yaml')
    assert 'val_ratio' in config['extraction']
    assert config['extraction']['val_ratio'] == 0.2
    assert isinstance(config['extraction']['val_ratio'], (float, int))
    assert 'split_seed' in config['extraction']
    assert config['extraction']['split_seed'] == 42
    assert isinstance(config['extraction']['split_seed'], int)