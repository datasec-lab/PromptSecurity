# attacks/loader.py
import importlib
import json
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

# Cache for loaded modules
_module_cache = {}

def load_attack(attack_name: str, **kwargs) -> Any:
    """动态加载攻击模块
    
    Args:
        attack_name: 攻击方法名称（文件夹名）
        **kwargs: 传递给攻击类的参数
        
    Returns:
        实例化的攻击对象
        
    Raises:
        ValueError: 当攻击模块不存在或无法加载时
    """
    # Handle special case
    if attack_name == "no_attack":
        from attacks.black_box.no_attack import NoAttack
        # NoAttack requires target_model
        if 'target_model' not in kwargs:
            raise ValueError("NoAttack requires 'target_model' parameter")
        return NoAttack(kwargs['target_model'])
    
    # Resolve attack path/module with compatibility aliases.
    normalized_name = attack_name.replace("-", "_")
    black_box_root = Path(__file__).parent / "black_box"
    white_box_root = Path(__file__).parent / "white_box"

    # Keep deterministic order: exact name first, then "XAttack" -> "X".
    candidate_names = [normalized_name]
    if normalized_name.lower().endswith("attack") and len(normalized_name) > len("attack"):
        candidate_names.append(normalized_name[:-len("attack")])

    attack_path = None
    module_path = None

    # 1) Direct path match
    for candidate in candidate_names:
        bb = black_box_root / candidate
        wb = white_box_root / candidate
        if bb.exists():
            attack_path = bb
            normalized_name = candidate
            module_path = f"attacks.black_box.{candidate}"
            break
        if wb.exists():
            attack_path = wb
            normalized_name = candidate
            module_path = f"attacks.white_box.{candidate}"
            break

    # 2) Case-insensitive fallback for convenience (e.g., lowercase CLI input)
    if attack_path is None:
        for candidate in candidate_names:
            lowered = candidate.lower()
            for item in black_box_root.iterdir():
                if item.is_dir() and item.name.lower() == lowered:
                    attack_path = item
                    normalized_name = item.name
                    module_path = f"attacks.black_box.{item.name}"
                    break
            if attack_path is not None:
                break
            for item in white_box_root.iterdir():
                if item.is_dir() and item.name.lower() == lowered:
                    attack_path = item
                    normalized_name = item.name
                    module_path = f"attacks.white_box.{item.name}"
                    break
            if attack_path is not None:
                break

    if attack_path is None or module_path is None:
        raise ValueError(f"Attack '{attack_name}' not found in black_box or white_box directories")

    # Check cache first
    cache_key = f"attacks.{normalized_name}"
    if cache_key in _module_cache:
        attack_class = _module_cache[cache_key]
    else:
        try:
            # Dynamic import
            module = importlib.import_module(module_path)
            
            # Find Attack class (convention)
            if hasattr(module, 'Attack'):
                attack_class = module.Attack
            else:
                # Backward compatibility: find class ending with 'Attack'
                attack_class = None
                for name, obj in vars(module).items():
                    if name.endswith('Attack') and isinstance(obj, type):
                        attack_class = obj
                        break
                
                if not attack_class:
                    raise ValueError(f"No Attack class found in {attack_name}")
            
            # Cache the class
            _module_cache[cache_key] = attack_class
            
        except ImportError as e:
            logger.error(f"Failed to import attack module {attack_name}: {e}")
            raise ValueError(f"Failed to import attack module {attack_name}: {e}")
    
    # Load default config if exists  
    if attack_path and (attack_path / "config.json").exists():
        config_file = attack_path / "config.json"
        try:
            with open(config_file, 'r') as f:
                default_config = json.load(f)
            # Extract parameters if nested
            if 'parameters' in default_config:
                default_params = default_config['parameters']
            else:
                default_params = default_config
            # Merge with user parameters
            params = {**default_params, **kwargs}
        except Exception as e:
            logger.warning(f"Failed to load config for {attack_name}: {e}")
            params = kwargs
    else:
        params = kwargs
    
    # Instantiate attack
    try:
        return attack_class(**params)
    except Exception as e:
        logger.error(f"Failed to instantiate attack {attack_name}: {e}")
        raise ValueError(f"Failed to instantiate attack {attack_name}: {e}")

def list_available_attacks() -> Dict[str, list]:
    """列出所有可用的攻击方法
    
    Returns:
        包含黑盒和白盒攻击列表的字典
    """
    attacks_dir = Path(__file__).parent
    
    black_box_attacks = []
    black_box_dir = attacks_dir / "black_box"
    if black_box_dir.exists():
        for item in black_box_dir.iterdir():
            if item.is_dir() and not item.name.startswith('__'):
                black_box_attacks.append(item.name)
    
    white_box_attacks = []
    white_box_dir = attacks_dir / "white_box"
    if white_box_dir.exists():
        for item in white_box_dir.iterdir():
            if item.is_dir() and not item.name.startswith('__'):
                white_box_attacks.append(item.name)
    
    # Add special case
    black_box_attacks.append("no_attack")
    
    return {
        "black_box": sorted(black_box_attacks),
        "white_box": sorted(white_box_attacks)
    }

def get_attack_info(attack_name: str) -> Dict[str, Any]:
    """获取攻击方法的详细信息
    
    Args:
        attack_name: 攻击方法名称
        
    Returns:
        包含攻击信息的字典
    """
    info = {
        "name": attack_name,
        "type": None,
        "config": None,
        "available": False
    }
    
    # Check type
    black_box_path = Path(__file__).parent / "black_box" / attack_name
    white_box_path = Path(__file__).parent / "white_box" / attack_name
    
    if black_box_path.exists():
        info["type"] = "black_box"
        config_path = black_box_path / "config.json"
    elif white_box_path.exists():
        info["type"] = "white_box"
        config_path = white_box_path / "config.json"
    else:
        return info
    
    info["available"] = True
    
    # Load config if exists
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                info["config"] = json.load(f)
        except:
            pass
    
    return info

def get_attack_config_path(attack_name: str) -> Optional[str]:
    """获取攻击配置文件路径
    
    Args:
        attack_name: 攻击方法名称
        
    Returns:
        配置文件路径或None（如果不存在）
    """
    # Handle special case
    if attack_name == "no_attack":
        return None
    
    # Check type
    black_box_path = Path(__file__).parent / "black_box" / attack_name
    white_box_path = Path(__file__).parent / "white_box" / attack_name
    
    if black_box_path.exists():
        config_path = black_box_path / "config.json"
    elif white_box_path.exists():
        config_path = white_box_path / "config.json"
    else:
        return None
    
    return str(config_path) if config_path.exists() else None
