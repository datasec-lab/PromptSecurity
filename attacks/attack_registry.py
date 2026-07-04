# attacks_outdated/attack_registry.py
from attacks.black_box.FlipAttack.FlipAttack import FlipAttack
from attacks.black_box.ArtPrompt.artprompt import ArtPromptAttack
from attacks.black_box.PAIR.pair_attack import PairAttack
from attacks.black_box.ABJAttack.abj_attack import ABJAttack
from attacks.black_box.PastTense.PastTenseAttack import PastTenseAttack
from attacks.black_box.IFSJ.ifsj_attack import IFSJAttack
from attacks.black_box.CodeAttack.CodeAttack import CodeAttack
from attacks.black_box.CodeChameleon.CodeChameleon import CodeChameleon
from attacks.black_box.DRA.DRA import DRA
from attacks.black_box.PersuasiveInContext.Persuasive_incontext_attack import PersuasiveInContextAttack
from attacks.black_box.ReNeLLM.ReNeLLM import ReNeLLM
from attacks.black_box.TapAttack.tapattack import TapAttack
from attacks.black_box.DrAttack.dr_attack import DrAttackAttack
from attacks.black_box.InceptionAttack.inception_attack import InceptionAttack
# Temporarily disabled due to import issues
# from attacks.black_box.MultilingualJailbreak.Multilingual_jailbreak_attack import MultilingualJailbreakAttack
MultilingualJailbreakAttack = None
from attacks.black_box.GPTFUZZER.GPTFUZZER import GPTFUZZER


from attacks.white_box.GCGAttack.GCGWhiteBoxAttack import GCGWhiteBoxAttack
from attacks.white_box.AutoDAN.auto_dan_attack import AutoDANAttack
from attacks.white_box.COLD.cold_attack import COLDAttack
from attacks.black_box.no_attack import NoAttack

ATTACKS = {
    #### Black-box Attacks ####
    'FlipAttack': FlipAttack,
    'ArtPromptAttack': ArtPromptAttack,
    'PairAttack':PairAttack,
    'ABJAttack': ABJAttack,
    'PastTenseAttack':PastTenseAttack,
    "IFSJAttack": IFSJAttack, # require LLM logits
    "CodeAttack": CodeAttack,
    'CodeChameleon': CodeChameleon,
    'DRA': DRA,
    'ReNeLLM': ReNeLLM,
    'InceptionAttack': InceptionAttack,
    'TapAttack': TapAttack,
    'DrAttackAttack': DrAttackAttack,
    # 'MultilingualJailbreakAttack': MultilingualJailbreakAttack,  # Temporarily disabled
    'GPTFUZZER': GPTFUZZER,
    'PersuasiveInContextAttack': PersuasiveInContextAttack,

    #### White-box Attacks ####
    'GCGWhiteBoxAttack': GCGWhiteBoxAttack,
    'AutoDANAttack': AutoDANAttack,
    'COLDAttack': COLDAttack,
    
    #### Baseline ####
    'no_attack': NoAttack,

}
