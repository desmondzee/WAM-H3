from fasterwam.datasets.lerobot.robot_video_dataset import RobotVideoDataset

from ..model.text_encoder import collate_instructions
from .text_cache import load_embedding


class WAMH3VideoDataset(RobotVideoDataset):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._text = {}

    def _get_cached_text_context(self, prompt):
        if prompt not in self._text:
            emb = load_embedding(self.text_embedding_cache_dir, prompt)
            if emb is None:
                raise FileNotFoundError(f"no cached embedding for {prompt!r}; run scripts/precompute_text_embeds.py")
            ctx, valid = collate_instructions([emb], self.context_len)
            self._text[prompt] = (ctx[0], valid[0])
        ctx, valid = self._text[prompt]
        return ctx.clone(), valid.clone()

    def _get(self, idx):
        s = super()._get(idx)
        s["context"], s["context_mask"] = self._get_cached_text_context(s["prompt"])
        s["proprio"] = s["proprio"][0]
        return s
