"""ACAC macro-action baseline with a parameter-matched decision head."""
from .acac import AgentCentricActorCritic
from acac.acac_marl.cores.acac_mlp.controller import MatchedMLPMAC

class MatchedMLPACAC(AgentCentricActorCritic):
    def __init__(self,env,*args,tg_embed_dim=32,mlp_head_width=0,
                 mlp_match_tolerance=0.01,**kwargs):
        kwargs["_controller_class"]=MatchedMLPMAC
        kwargs["_controller_options"]=dict(tg_embed_dim=tg_embed_dim,
            mlp_head_width=mlp_head_width,mlp_match_tolerance=mlp_match_tolerance)
        super().__init__(env,*args,**kwargs)
