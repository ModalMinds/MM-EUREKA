from ..base.patch import BasePatch
import torch

class Qwen2_5_VLPatch(BasePatch):
    def _add_get_inputs_embeds():
        from transformers import Qwen2_5_VLForConditionalGeneration
        def get_inputs_embeds(self, input_ids, image_grid_thw=None, video_grid_thw=None, pixel_values=None, pixel_values_videos=None, **kwargs):
            inputs_embeds = self.model.embed_tokens(input_ids)
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.dtype)
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
                n_image_features = image_embeds.shape[0]
                if n_image_tokens != n_image_features:
                    raise ValueError(
                        f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                    )

                mask = input_ids == self.config.image_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                image_mask = mask_expanded.to(inputs_embeds.device)

                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
                video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
                n_video_features = video_embeds.shape[0]
                if n_video_tokens != n_video_features:
                    raise ValueError(
                        f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                    )

                mask = input_ids == self.config.video_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                video_mask = mask_expanded.to(inputs_embeds.device)

                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)
            return inputs_embeds

        Qwen2_5_VLForConditionalGeneration.get_inputs_embeds = get_inputs_embeds

    def _add_get_position_ids():
        from transformers import Qwen2_5_VLForConditionalGeneration
        def get_position_ids(self, input_ids, image_grid_thw=None, video_grid_thw=None, attention_mask=None, **kwargs):
            position_ids,mrope_position_deltas = self.get_rope_index(input_ids=input_ids, image_grid_thw=image_grid_thw, video_grid_thw=video_grid_thw, attention_mask=attention_mask)
            return position_ids
        Qwen2_5_VLForConditionalGeneration.get_position_ids = get_position_ids

    def _add_offset_split_position_ids():
        from transformers import Qwen2_5_VLForConditionalGeneration
        def offset_split_position_ids(self,position_ids,hacked_position_ids):
            new_position_ids = position_ids.clone()
            for i in range(hacked_position_ids.size(0)):
                seq_idxes = torch.nonzero(hacked_position_ids[i]==0)[:,0]
                seq_idxes = torch.cat([seq_idxes, torch.tensor([hacked_position_ids.size(1)],device=seq_idxes.device)], dim=0)
                st = 0
                for seq_idx in seq_idxes:
                    if st == 0 and seq_idx == 0:
                        continue
                    #shape: [3,bs,seq_len]
                    raw_seq_position_ids = position_ids[:,i,st:seq_idx]
                    new_position_ids[:,i,st:seq_idx] = raw_seq_position_ids - raw_seq_position_ids[:,:1] + hacked_position_ids[i,st]
                    st = seq_idx
            return new_position_ids
        Qwen2_5_VLForConditionalGeneration.offset_split_position_ids = offset_split_position_ids
    
    @classmethod
    def _load_all_patches(cls):
        cls._add_get_inputs_embeds()
        cls._add_get_position_ids()
        cls._add_offset_split_position_ids()

Patch = Qwen2_5_VLPatch()