from tools.inverse_design import MirochannelDesignSearcher
from config.config import parse_args, create_cfg_from_args

args = parse_args()
cfg = create_cfg_from_args(args)
if not cfg.eval_cfg.infer_weight_path:
    cfg.eval_cfg.infer_weight_path = r'../log/CEyeNet/CEyeNet'
searcher = MirochannelDesignSearcher(cfg)
searcher()