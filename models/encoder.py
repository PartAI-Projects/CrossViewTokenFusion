import torch
import torch.nn as nn
import torch.nn.functional as F


class VisionEncoderWithDeepPrompts(nn.Module):
    def __init__(self, medsiglip_model, deep_prompt):
        super().__init__()
        self.vision_model = medsiglip_model.vision_model
        self.deep_prompt = deep_prompt
        
        # Pretrained head from MedSigLIP
        self.head = self.vision_model.head
                
        # Freeze backbone
        for p in self.vision_model.parameters():
            p.requires_grad = False
        
        # Freeze the pretrained head
        for p in self.head.parameters():
            p.requires_grad = False
            
        print("✓ Vision encoder FROZEN")
        print("✓ Pretrained head loaded and FROZEN")
    
    def forward(self, pixel_values):
        bsz = pixel_values.size(0)
        # Original patch + position embeddings
        patch_embeddings = self.vision_model.embeddings(pixel_values)  
        shallow_prompt = self.deep_prompt.get_shallow()
        # Expand Shallow prompt for batch
        sh_prompt = shallow_prompt.expand(bsz, -1, -1)                       
        # Add Shallow prompt in front of patch tokens
        hidden_states = torch.cat([sh_prompt, patch_embeddings], dim=1)      
        
        # Forward through transformer layers with deep prompts
        for layer_idx, layer in enumerate(self.vision_model.encoder.layers):
            # Get deep prompt for this layer
            deep_p = self.deep_prompt.get_deep_prompt(layer_idx)
            if deep_p is not None:
                deep_b = deep_p.unsqueeze(0).expand(bsz, -1, -1)      
                # replace first token with deep prompt
                hidden_states[:, 0:1, :] = deep_b
            hidden_states = layer(hidden_states, attention_mask=None)[0]
        
        # Final layer norm
        hidden_states = self.vision_model.post_layernorm(hidden_states)
        # Apply pretrained head (multihead attention pooling)
        pooled_output = self.head(hidden_states)                            
        # Normalize features
        feats = F.normalize(pooled_output, dim=-1)
        return feats


class VisionEncoderWithFusionBlocks(nn.Module):
    def __init__(self, medsiglip_model, deep_prompt):
        super().__init__()
        self.vision_model = medsiglip_model.vision_model
        self.deep_prompt = deep_prompt
        self.head = self.vision_model.head
        self.num_layers = len(self.vision_model.encoder.layers)

        for p in self.vision_model.parameters():
            p.requires_grad = False
        for p in self.head.parameters():
            p.requires_grad = True
            
        print(f"✓ Vision encoder FROZEN - {self.num_layers} layers total")
    
    def encode_up_to_layer(self, pixel_values, stop_layer):
        """
        Encode up to specified layer (for fusion point)
        
        Args:
            pixel_values: input images
            stop_layer: stop after this layer (0-indexed)
        
        Returns:
            hidden_states after stop_layer
        """
        bsz = pixel_values.size(0)
        patch_embeddings = self.vision_model.embeddings(pixel_values)
        shallow_prompt = self.deep_prompt.get_shallow()
        sh_prompt = shallow_prompt.expand(bsz, -1, -1)
        hidden_states = torch.cat([sh_prompt, patch_embeddings], dim=1)

        # Forward through layers 0 to stop_layer
        layers = self.vision_model.encoder.layers
        for layer_idx in range(stop_layer + 1):
            layer = layers[layer_idx]
            # Inject deep prompt (only for layers with prompts)
            deep_p = self.deep_prompt.get_deep_prompt(layer_idx)
            if deep_p is not None:
                deep_b = deep_p.unsqueeze(0).expand(bsz, -1, -1)
                hidden_states[:, 0:1, :] = deep_b
            hidden_states = layer(hidden_states, attention_mask=None)[0]
        return hidden_states
    
    def continue_from_layer(self, hidden_states, start_layer, stop_layer):
        """
        Continue encoding from start_layer to stop_layer
        
        Args:
            hidden_states: fused hidden states
            start_layer: resume from this layer (0-indexed)
            stop_layer: continue untill this layer (0-indexed)
        Returns:
            hidden_states after stop_layer
        """
        layers = self.vision_model.encoder.layers
        for layer_idx in range(start_layer+1, stop_layer+1):
            layer = layers[layer_idx]
            hidden_states = layer(hidden_states, attention_mask=None)[0] 
        return hidden_states
        
    def continue_to_end(self, hidden_states, start_layer):
        """
        Continue encoding from start_layer to end
        
        Args:
            hidden_states: fused hidden states
            start_layer: resume from this layer (0-indexed)
        
        Returns:
            feats: final normalized features
        """
        layers = self.vision_model.encoder.layers
        # Continue through remaining layers (start_layer+1 to end)
        for layer_idx in range(start_layer+1, len(layers)):
            layer = layers[layer_idx]
            hidden_states = layer(hidden_states, attention_mask=None)[0]
    
        # Final layer norm
        hidden_states = self.vision_model.post_layernorm(hidden_states)
        # Apply pretrained head
        pooled_output = self.head(hidden_states)
        # Normalize features
        feats = F.normalize(pooled_output, dim=-1)
        return feats