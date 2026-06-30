import torch
import torch.nn as nn


class DeepPrompt(nn.Module):
    def __init__(
        self,
        cfg,
        hidden_size,
        device="cuda",
        dtype=torch.float32,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.prompt_depth = cfg.model.prompt_depth
        self.num_prompt_tokens = cfg.model.shallow_prompt_length

        self.shallow_prompt = nn.Parameter(
            torch.empty(
                1,
                self.num_prompt_tokens,
                self.hidden_size,
                dtype=dtype,
                device=device,
            )
        )
        nn.init.normal_(
            self.shallow_prompt,
            std=self.hidden_size ** -0.5,
        )

        self.deep_prompts = nn.ParameterList(
            [
                nn.Parameter(
                    torch.empty(
                        1,
                        self.hidden_size,
                        dtype=dtype,
                        device=device,
                    )
                )
                for _ in range(self.prompt_depth)
            ]
        )
        for p in self.deep_prompts:
            nn.init.normal_(p, std=0.02)

    def get_shallow(self):
        return self.shallow_prompt

    def get_deep_prompt(self, layer_idx):
        if layer_idx >= self.prompt_depth:
            return None
        return self.deep_prompts[layer_idx]