import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
import os
import json
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

class OpenCLIPTextEncoder(nn.Module):
    def __init__(self, model_name="ViT-B-16", pretrained='laion2b_s34b_b88k'):
        super().__init__()
        import open_clip
        result = open_clip.create_model_and_transforms(
            model_name=model_name,
            pretrained=pretrained,
            output_dict=True
        )
        if len(result) == 4:
            model, preprocess_train, preprocess_val, _ = result
            preprocess = preprocess_train
        else:
            model, preprocess, _ = result
        self.text_model = model.transformer
        self.token_embedding = model.token_embedding
        self.positional_embedding = model.positional_embedding
        self.ln_final = model.ln_final
        self.text_projection = model.text_projection
        import open_clip
        self.tokenizer = open_clip.tokenize
        self.preprocess = preprocess
        self.context_length = model.context_length

        for param in self.text_model.parameters():
            param.requires_grad = False
        for param in self.token_embedding.parameters():
            param.requires_grad = False
        if self.positional_embedding is not None:
            self.positional_embedding.requires_grad = False
        for param in self.ln_final.parameters():
            param.requires_grad = False
        if self.text_projection is not None:
            if isinstance(self.text_projection, nn.Module):
                for param in self.text_projection.parameters():
                    param.requires_grad = False
            else:
                self.text_projection.requires_grad = False

    def forward(self, text):
        x = self.token_embedding(text)
        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)
        x = self.text_model(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x

class OpenCLIPVisionEncoder(nn.Module):
    def __init__(self, model_name="ViT-B-16", pretrained="laion2b_s34b_b88k", unfreeze_last_n_layers=0):
        super().__init__()
        import open_clip

        model, preprocess, _ = open_clip.create_model_and_transforms(
            model_name=model_name,
            pretrained=pretrained,
            output_dict=True
        )
        

        self.visual = model.visual
        self.preprocess = preprocess
        

        for param in self.visual.parameters():
            param.requires_grad = False
        print(f"OpenCLIP Vision Encoder ({model_name}, {pretrained}) loaded and frozen.")
        

        if unfreeze_last_n_layers > 0:
            if hasattr(self.visual, 'transformer') and hasattr(self.visual.transformer, 'resblocks'):
                all_layers = self.visual.transformer.resblocks
                num_total_layers = len(all_layers)
                layers_to_unfreeze = min(unfreeze_last_n_layers, num_total_layers)
                print(f"Detected {num_total_layers} Transformer blocks. Unfreezing last {layers_to_unfreeze} blocks...")
                for layer in all_layers[-layers_to_unfreeze:]:
                    for param in layer.parameters():
                        param.requires_grad = True
                print(f"Successfully unfroze last {layers_to_unfreeze} block parameters for fine-tuning.")
            else:
                print("Warning: Cannot find transformer.resblocks structure in OpenCLIP visual model.")
        

        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            dummy_output = self.visual(dummy_input)
            self.output_dim = dummy_output.shape[-1]
    
    def forward(self, x):

        return self.visual(x)
    
    def get_patch_features(self, x):


        x = self.visual.conv1(x) 
        x = x.reshape(x.shape[0], x.shape[1], -1)  
        x = x.permute(0, 2, 1)  
        

        x = torch.cat([self.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
        x = x + self.visual.positional_embedding.to(x.dtype)
        x = self.visual.ln_pre(x)
        
        x = x.permute(1, 0, 2)  
        x = self.visual.transformer(x)
        x = x.permute(1, 0, 2)  
        

        return x[:, 1:, :]  

class ProjectionHead(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, output_dim)
        )
    def forward(self, x):
        return self.proj(x)

class Generator_MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=1024, num_layers=6):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.GELU()]
        for _ in range(num_layers - 2):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU()])
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.model = nn.Sequential(*layers)
        self.residual = (input_dim == output_dim)
    def forward(self, x):
        out = self.model(x)
        if self.residual:
            return out + x
        return out

class Discriminator_MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=1024, num_layers=6):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.GELU()]
        for _ in range(num_layers - 2):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU()])
        layers.append(nn.Linear(hidden_dim, 1))
        self.model = nn.Sequential(*layers)
        self.residual = False
    def forward(self, x):
        out = self.model(x)
        return out

def r1_gradient_penalty(d_real_out, real_data):

    grad_real, = torch.autograd.grad(
        outputs=d_real_out.sum(), inputs=real_data, create_graph=True
    )
    grad_penalty = grad_real.pow(2).view(grad_real.shape[0], -1).sum(1).mean()
    return grad_penalty

def nt_xent_loss(embeddings1, embeddings2, temperature=0.07):
    embeddings1 = F.normalize(embeddings1, dim=1)
    embeddings2 = F.normalize(embeddings2, dim=1)
    similarity_matrix = torch.matmul(embeddings1, embeddings2.T)
    similarity_matrix = similarity_matrix / temperature
    labels = torch.arange(similarity_matrix.size(0)).to(embeddings1.device)
    loss = F.cross_entropy(similarity_matrix, labels)
    return loss

def nt_xent_loss_bidirectional(embeddings1, embeddings2, temperature=0.07):
    loss1 = nt_xent_loss(embeddings1, embeddings2, temperature)
    loss2 = nt_xent_loss(embeddings2, embeddings1, temperature)
    return (loss1 + loss2) / 2

class SatelliteTextAlignmentDataset(Dataset):
    def __init__(self, json_path=None, transform=None, tokenizer=None, image_base_dir=None, context_length=77, data_items=None):
        super().__init__()
        self.transform = transform
        self.tokenizer = tokenizer
        self.image_base_dir = image_base_dir
        self.context_length = context_length
        if data_items is not None:
            self.data_items = data_items
        else:
            self.data_items = []
            with open(json_path, 'r') as f:
                full_json_data = json.load(f)
            for image_caption_list in full_json_data:
                if not image_caption_list: continue
                image_path_rel = image_caption_list[0]["image"]
                random_caption_dict = np.random.choice(image_caption_list)
                caption = random_caption_dict["caption"]
                full_image_path = image_path_rel
                self.data_items.append({
                    "image_path": full_image_path,
                    "caption": caption
                })
        print(f"SatelliteTextAlignmentDataset loaded {len(self.data_items)} items.")

    def __len__(self):
        return len(self.data_items)

    def __getitem__(self, idx):
        item = self.data_items[idx]
        image_path = item["image_path"]
        caption = item["caption"]
        try:
            image_pil = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image_pil)
        except Exception as e:
            print(f"Warning: Error loading image {image_path}: {e}. Skipping this sample.")
            return None, None
        tokenized_text = self.tokenizer([caption], context_length=self.context_length)[0]
        return image_tensor, tokenized_text

class CVUSACorrespondenceDatasetUpdated(Dataset):
    def __init__(self, df_subset, base_config,
                 transform_sat,
                 transform_street_crop,
                 text_json_path=None,
                 tokenizer=None
                 ):
        super().__init__()
        self.df = df_subset.reset_index(drop=True)
        self.base_config = base_config
        self.transform_sat = transform_sat
        self.transform_street_crop = transform_street_crop
        self.num_street_crops = base_config.get('num_street_crops', 1)
        self.crop_size = base_config['image_size']
        
        self.text_data = {}
        if text_json_path and tokenizer:
            with open(text_json_path, 'r') as f:
                full_json_data = json.load(f)
            if len(full_json_data) > 0:
                for image_caption_list in full_json_data:
                    if image_caption_list:
                        rel_path = os.path.relpath(image_caption_list[0]["image"], "./data")
                        self.text_data[rel_path] = [item["caption"] for item in image_caption_list]
            self.tokenizer = tokenizer
        else:
            self.tokenizer = None

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        filename_col_sat = self.base_config.get('csv_file_col_names', ['sat_path', 'street_path'])[0]
        filename_col_street = self.base_config.get('csv_file_col_names', ['sat_path', 'street_path'])[1]
        sat_img_p = str(self.df.iloc[idx][filename_col_sat])
        street_img_p = str(self.df.iloc[idx][filename_col_street])
        
        if not self.base_config['csv_path_is_absolute']:
            sat_img_path = os.path.join(self.base_config['bingmap_dir'], sat_img_p)
            street_img_path = os.path.join(self.base_config['streetview_dir'], street_img_p)
        else: 
            sat_img_path = sat_img_p; street_img_path = street_img_p

        text_tokens = None
        try:
            sat_image_pil = Image.open(sat_img_path).convert('RGB')
            sat_image_tensor = self.transform_sat(sat_image_pil) if self.transform_sat else transforms.ToTensor()(sat_image_pil)

            street_image_pil = Image.open(street_img_path).convert('RGB')
            original_w, original_h = street_image_pil.size
            street_crops_tensors = []
            if self.num_street_crops == 1:
                temp_transform = transforms.Compose([transforms.Resize(self.crop_size), transforms.CenterCrop(self.crop_size)])
                street_image_pil_resized_cropped = temp_transform(street_image_pil)
                if self.transform_street_crop: street_crops_tensors.append(self.transform_street_crop(street_image_pil_resized_cropped))
                else: street_crops_tensors.append(transforms.ToTensor()(street_image_pil_resized_cropped))
            elif self.num_street_crops > 1:
                crop_w = min(self.crop_size, original_w)
                start_xs = np.linspace(0, original_w - crop_w, self.num_street_crops, dtype=int)
                for start_x in start_xs:
                    left, top, right, bottom = start_x, (original_h - self.crop_size) // 2, start_x + crop_w, (original_h - self.crop_size) // 2 + self.crop_size
                    crop_pil = street_image_pil.crop((left, top, right, bottom))
                    crop_pil = transforms.Resize((self.crop_size, self.crop_size))(crop_pil)
                    if self.transform_street_crop: street_crops_tensors.append(self.transform_street_crop(crop_pil))
                    else: street_crops_tensors.append(transforms.ToTensor()(crop_pil))
            else: raise ValueError(f"num_street_crops must be >= 1")
            
            if not street_crops_tensors: return None, None, None

            final_street_tensor = torch.stack(street_crops_tensors)
            
            if self.text_data and self.tokenizer:
                rel_path = os.path.relpath(sat_img_path, self.base_config['bingmap_dir'])
                text_captions_for_image = self.text_data.get(rel_path)
                if text_captions_for_image:
                    selected_caption = np.random.choice(text_captions_for_image)
                    text_tokens = self.tokenizer([selected_caption])[0]
                else:
                    print(f"No text found: key={rel_path}")
            
            return sat_image_tensor, final_street_tensor, text_tokens
        except Exception as e: 
            print(f"Warning: Error loading index {idx} ({sat_img_path}...): {e}"); return None, None, None

CONFIG_R3GAN_STYLE = {
    "cvusa_csv_path": "./data/CVUSA_subset/file_paths.csv",
    "bingmap_dir": "./data",
    "streetview_dir": "./data",
    "csv_path_is_absolute": False,
    "csv_file_col_names": ['sat_path', 'street_path'],

    "image_size": 224,
    "num_street_crops": 5,
    "validation_split_ratio": 0.1,
    "random_state_split": 42,


    "vision_encoder_model_name": "ViT-B-16",
    "vision_encoder_pretrained": "laion2b_s34b_b88k",  
    "vision_encoder_unfreeze_layers": 0,
    
    "text_encoder_model_name": "ViT-B-16",
    "text_encoder_pretrained": "laion2b_s34b_b88k",
    "text_encoder_unfreeze_layers": 0,

    "cvusa_text_json_path": "./data/CVUSA_bingmapcaptions_output.json",
    "json_image_base_dir": "./data/CVUSA_subset/bingmap/",
    "g_hidden_dim": 1024, "g_num_layers": 4,
    "d_hidden_dim": 1024, "d_num_layers": 4,

    "batch_size": 128,
    "lr_g_sat_encoder": 5e-5,  
    "lr_d": 5e-5,
    "lr_text_encoder": 1e-6,
    "lr_sat_encoder_stage1": 1e-6,
    "lr_proj_head": 5e-5,  
    "beta1_adam": 0.0,
    "beta2_adam": 0.99,
    "lambda_r1": 40,

    "num_epochs_stage1_text_align": 1,
    "num_epochs_stage2_r3gan": 1,

    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),

    "projection_dim": 512,
    "lambda_fine": 1.0,
    "sat_encoder_stage1_save_path": "sat_encoder_stage1_aligned_freeze.pth",
    "text_encoder_stage1_save_path": "text_encoder_stage1_aligned_freeze.pth",
    "sat_projection_head_stage1_save_path": "sat_projection_head_stage1_freeze.pth",
    "text_projection_head_stage1_save_path": "text_projection_head_stage1_freeze.pth",
    "overall_best_sat_encoder_save_path": "r3gan_style_sat_encoder_best_freeze.pth",
    "overall_best_generator_save_path": "r3gan_style_generator_best_freeze.pth",


    "lambda_FM": 50.0,  
    "contrastive_temperature_init": 0.07, 
}

def fine_grained_contrastive_loss(patch_feats, token_feats, temperature=0.07):


    patch_feats = F.normalize(patch_feats, dim=-1)
    token_feats = F.normalize(token_feats, dim=-1)
    
    batch_size = patch_feats.size(0)
    

    sim_matrix = torch.einsum('bnd,bmd->bnm', patch_feats, token_feats)
    

    max_sim_i2t, _ = sim_matrix.max(dim=2)
    s_i2t = max_sim_i2t.mean(dim=1)
    

    max_sim_t2i, _ = sim_matrix.max(dim=1)
    s_t2i = max_sim_t2i.mean(dim=1)

    s_i2t = s_i2t / temperature
    s_t2i = s_t2i / temperature
    

    exp_s_i2t = torch.exp(s_i2t)
    sum_exp_i2t = exp_s_i2t.sum()
    loss_i2t = 0.0
    for i in range(batch_size):
        pos_score = exp_s_i2t[i]
        loss_i2t += -torch.log(pos_score / sum_exp_i2t)
    loss_i2t = loss_i2t / batch_size
  
    exp_s_t2i = torch.exp(s_t2i)
    sum_exp_t2i = exp_s_t2i.sum()
    loss_t2i = 0.0
    for i in range(batch_size):
        pos_score = exp_s_t2i[i]
        loss_t2i += -torch.log(pos_score / sum_exp_t2i)
    loss_t2i = loss_t2i / batch_size
    

    loss_total = 0.5 * (loss_i2t + loss_t2i)
    
    return loss_total

def collate_fn_skip_none(batch):
    filtered_batch = []
    for item in batch:
        if item is None: continue
        is_valid_item = True
        for element in item:
            if element is None:
                is_valid_item = False
                break
            if isinstance(element, torch.Tensor) and element.numel() == 0:
                is_valid_item = False
                break
        if is_valid_item:
            filtered_batch.append(item)

    if not filtered_batch: return None
    return torch.utils.data.dataloader.default_collate(filtered_batch)



def train_satellite_text_alignment():
    config = CONFIG_R3GAN_STYLE
    device = config['device']
    print(f"\n--- Stage 1: Satellite-Text Alignment Training ---")


    
    sat_encoder = OpenCLIPVisionEncoder(
        model_name=config['vision_encoder_model_name'],
        pretrained=config['vision_encoder_pretrained'],
        unfreeze_last_n_layers=config['vision_encoder_unfreeze_layers']
    ).to(device)


    text_encoder_instance = OpenCLIPTextEncoder(
        model_name=config['text_encoder_model_name'],
        pretrained=config['text_encoder_pretrained']
    ).to(device)
    

    if config['text_encoder_unfreeze_layers'] > 0:
        if hasattr(text_encoder_instance.text_model, 'resblocks'):
            all_layers = text_encoder_instance.text_model.resblocks
            num_total_layers = len(all_layers)
            layers_to_unfreeze = min(config['text_encoder_unfreeze_layers'], num_total_layers)
            for layer in all_layers[-layers_to_unfreeze:]:
                for param in layer.parameters():
                    param.requires_grad = True
            print(f"Successfully unfroze last {layers_to_unfreeze} Transformer block parameters for fine-tuning.")
        else:
            print("Warning: Cannot find 'resblocks' structure in text encoder, cannot perform partial unfreezing.")
        

        for param in text_encoder_instance.token_embedding.parameters():
            param.requires_grad = True
        if text_encoder_instance.positional_embedding is not None:
             text_encoder_instance.positional_embedding.requires_grad = True
        for param in text_encoder_instance.ln_final.parameters():
            param.requires_grad = True
        if text_encoder_instance.text_projection is not None:
            if isinstance(text_encoder_instance.text_projection, nn.Module):
                for param in text_encoder_instance.text_projection.parameters():
                    param.requires_grad = True
            else:
                text_encoder_instance.text_projection.requires_grad = True



    sat_encoder_output_dim = sat_encoder.output_dim
    

    with torch.no_grad():
        import open_clip
        dummy_text = ["hello world"]
        dummy_tokenized = open_clip.tokenize(dummy_text)
        model_max_length = dummy_tokenized.shape[1]
        vocab_size = text_encoder_instance.token_embedding.weight.shape[0]
        dummy_text_input = torch.randint(0, vocab_size, (1, model_max_length)).to(device)
        dummy_text_output = text_encoder_instance(dummy_text_input)
        text_encoder_output_dim = dummy_text_output.shape[-1]
    print(f"Satellite encoder output dimension: {sat_encoder_output_dim}")
    print(f"Text encoder output dimension: {text_encoder_output_dim}")

    sat_projection_head = ProjectionHead(sat_encoder_output_dim, 768).to(device)
    text_projection_head = ProjectionHead(text_encoder_output_dim, 768).to(device)
    sat_patch_projection_head = ProjectionHead(768, 768).to(device)
    print(f"Both satellite and text projection heads map features to 768 dimensions.")


    transform_sat = transforms.Compose([
        transforms.Resize((config['image_size'], config['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    import random
    with open(config['cvusa_text_json_path'], 'r') as f:
        full_json_data = json.load(f)
    total_len = len(full_json_data)
    indices = list(range(total_len))
    random.seed(config['random_state_split'])
    random.shuffle(indices)
    val_size = int(total_len * config['validation_split_ratio'])
    val_indices = set(indices[:val_size])
    train_indices = set(indices[val_size:])
    train_json = [full_json_data[i] for i in train_indices]
    val_json = [full_json_data[i] for i in val_indices]

    def build_data_items(json_data):
        items = []
        for image_caption_list in json_data:
            if not image_caption_list: continue
            image_path_rel = image_caption_list[0]["image"]
            random_caption_dict = np.random.choice(image_caption_list)
            caption = random_caption_dict["caption"]
            full_image_path = image_path_rel
            items.append({
                "image_path": full_image_path,
                "caption": caption
            })
        return items

    train_data_items = build_data_items(train_json)
    val_data_items = build_data_items(val_json)

    train_dataset = SatelliteTextAlignmentDataset(
        json_path=None,
        transform=text_encoder_instance.preprocess,
        tokenizer=open_clip.tokenize,
        image_base_dir=config['json_image_base_dir'],
        context_length=text_encoder_instance.context_length,
        data_items=train_data_items
    )
    val_dataset = SatelliteTextAlignmentDataset(
        json_path=None,
        transform=text_encoder_instance.preprocess,
        tokenizer=open_clip.tokenize,
        image_base_dir=config['json_image_base_dir'],
        context_length=text_encoder_instance.context_length,
        data_items=val_data_items
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn_skip_none
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn_skip_none
    )


    temperature = nn.Parameter(torch.tensor(config['contrastive_temperature_init'], device=device))
    print(f"Initialized learnable temperature parameter: {temperature.item():.4f}")

    optimizer_stage1 = optim.Adam([
        {'params': filter(lambda p: p.requires_grad, sat_encoder.parameters()), 'lr': config['lr_sat_encoder_stage1']},
        {'params': filter(lambda p: p.requires_grad, text_encoder_instance.parameters()), 'lr': config['lr_text_encoder']},
        {'params': sat_projection_head.parameters(), 'lr': config['lr_proj_head']},
        {'params': text_projection_head.parameters(), 'lr': config['lr_proj_head']},
        {'params': sat_patch_projection_head.parameters(), 'lr': config['lr_proj_head']},
        {'params': [temperature], 'lr': config['lr_proj_head']},
    ], betas=(config['beta1_adam'], config['beta2_adam']))

    best_val_loss_stage1 = float('inf')
    print(f"Starting Stage 1: Satellite-Text Alignment Training ({config['num_epochs_stage1_text_align']} Epochs)...")
    for epoch in range(config['num_epochs_stage1_text_align']):
        sat_encoder.train()
        text_encoder_instance.train()
        sat_projection_head.train()
        text_projection_head.train()
        sat_patch_projection_head.train()

        total_loss = 0
        num_batches = 0
        for i, data_batch in enumerate(train_loader):
            if data_batch is None: continue
            sat_imgs, text_tokens = data_batch
            if sat_imgs is None or text_tokens is None: continue

            sat_imgs = sat_imgs.to(device)
            text_tokens = text_tokens.to(device)

            optimizer_stage1.zero_grad()

            sat_features = sat_encoder(sat_imgs)
            text_features = text_encoder_instance(text_tokens)
            sat_features_projected = sat_projection_head(sat_features)
            text_features_projected = text_projection_head(text_features)
            loss_global = nt_xent_loss_bidirectional(sat_features_projected, text_features_projected, temperature=temperature)

            with torch.no_grad():
                patch_feats = sat_encoder.get_patch_features(sat_imgs)

            B, N, D = patch_feats.shape
            patch_feats_flat = patch_feats.reshape(B * N, D)
            patch_feats_proj = sat_patch_projection_head(patch_feats_flat)
            patch_feats_proj = patch_feats_proj.reshape(B, N, -1)
            with torch.no_grad():
                text_emb = text_encoder_instance.token_embedding(text_tokens) + text_encoder_instance.positional_embedding
                text_feats = text_encoder_instance.text_model(text_emb.permute(1, 0, 2)).permute(1, 0, 2)
            text_feats_proj = text_projection_head(text_feats)
            loss_fine = fine_grained_contrastive_loss(patch_feats_proj, text_feats_proj, temperature=temperature)

            loss = loss_global + config.get('lambda_fine', 1.0) * loss_fine

            loss.backward()
            optimizer_stage1.step()

            total_loss += loss.item()
            num_batches += 1

            if i % 100 == 0:
                print(f"Stage 1 Epoch {epoch+1}/{config['num_epochs_stage1_text_align']} Batch {i}/{len(train_loader)} Loss: {loss.item():.4f} (Global: {loss_global.item():.4f} Fine: {loss_fine.item():.4f})")

        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        print(f"Stage 1 Epoch {epoch+1} Train Average Loss: {avg_loss:.4f}")
        sat_encoder.eval()
        text_encoder_instance.eval()
        sat_projection_head.eval()
        text_projection_head.eval()
        sat_patch_projection_head.eval()
        val_total_loss = 0
        val_num_batches = 0
        with torch.no_grad():
            for val_batch in val_loader:
                if val_batch is None: continue
                val_imgs, val_text_tokens = val_batch
                if val_imgs is None or val_text_tokens is None: continue
                val_imgs = val_imgs.to(device)
                val_text_tokens = val_text_tokens.to(device)
                val_sat_features = sat_encoder(val_imgs)
                val_text_features = text_encoder_instance(val_text_tokens)
                val_sat_features_projected = sat_projection_head(val_sat_features)
                val_text_features_projected = text_projection_head(val_text_features)
                val_loss_global = nt_xent_loss_bidirectional(val_sat_features_projected, val_text_features_projected, temperature=temperature)
                patch_feats = sat_encoder.get_patch_features(val_imgs)

                B, N, D = patch_feats.shape
                patch_feats_flat = patch_feats.reshape(B * N, D)
                patch_feats_proj = sat_patch_projection_head(patch_feats_flat)
                patch_feats_proj = patch_feats_proj.reshape(B, N, -1)
                text_emb = text_encoder_instance.token_embedding(val_text_tokens) + text_encoder_instance.positional_embedding
                text_feats = text_encoder_instance.text_model(text_emb.permute(1, 0, 2)).permute(1, 0, 2)
                text_feats_proj = text_projection_head(text_feats)
                val_loss_fine = fine_grained_contrastive_loss(patch_feats_proj, text_feats_proj, temperature=temperature)
                val_loss = val_loss_global + config.get('lambda_fine', 1.0) * val_loss_fine
                val_total_loss += val_loss.item()
                val_num_batches += 1
        avg_val_loss = val_total_loss / val_num_batches if val_num_batches > 0 else 0
        print(f"Stage 1 Epoch {epoch+1} Val Average Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss_stage1:
            best_val_loss_stage1 = avg_val_loss
            torch.save(sat_encoder.state_dict(), config['sat_encoder_stage1_save_path'])
            torch.save(text_encoder_instance.state_dict(), config['text_encoder_stage1_save_path'])
            torch.save(sat_projection_head.state_dict(), config['sat_projection_head_stage1_save_path'])
            torch.save(text_projection_head.state_dict(), config['text_projection_head_stage1_save_path'])
            torch.save(sat_patch_projection_head.state_dict(), 'sat_patch_projection_head_stage1.pth')
            print(f"  ==> Stage 1 new best validation loss: {best_val_loss_stage1:.6f}. Best model saved.")
    print("Stage 1: Satellite-Text Alignment Training completed!")


def train_r3gan_style_stage2():
    import open_clip
    config = CONFIG_R3GAN_STYLE
    device = config['device']
    print(f"\n--- Stage 2: R3GAN Generator Training ---")
    print(f"Using device: {device}")

    try:
        temp_vision_encoder = OpenCLIPVisionEncoder(
            model_name=config['vision_encoder_model_name'],
            pretrained=config['vision_encoder_pretrained']
        )
        openclip_native_dim = temp_vision_encoder.output_dim
        del temp_vision_encoder
        print(f"Detected OpenCLIP ViT-B/16 native feature dimension: {openclip_native_dim}")
    except Exception as e: 
        print(f"Error: {e}"); return


    sat_encoder = OpenCLIPVisionEncoder(
        model_name=config['vision_encoder_model_name'],
        pretrained=config['vision_encoder_pretrained'],
        unfreeze_last_n_layers=0
    ).to(device)
    if os.path.exists(config['sat_encoder_stage1_save_path']):
        sat_encoder.load_state_dict(torch.load(config['sat_encoder_stage1_save_path'], map_location=device))
        print(f"Loaded Stage 1 trained satellite encoder weights: {config['sat_encoder_stage1_save_path']}")
    else:
        print(f"Warning: Stage 1 satellite encoder weights not found: {config['sat_encoder_stage1_save_path']}. Using default initialization.")
    sat_encoder.eval()
    sat_projection_head = ProjectionHead(openclip_native_dim, 768).to(device)
    text_projection_head = ProjectionHead(512, 768).to(device)
    if os.path.exists(config['sat_projection_head_stage1_save_path']):
        sat_projection_head.load_state_dict(torch.load(config['sat_projection_head_stage1_save_path'], map_location=device))
        print(f"Loaded Stage 1 satellite projection head weights: {config['sat_projection_head_stage1_save_path']}")
    else:
        print(f"Warning: Stage 1 satellite projection head weights not found: {config['sat_projection_head_stage1_save_path']}. Using default initialization.")
    if os.path.exists(config['text_projection_head_stage1_save_path']):
        text_projection_head.load_state_dict(torch.load(config['text_projection_head_stage1_save_path'], map_location=device))
        print(f"Loaded Stage 1 text projection head weights: {config['text_projection_head_stage1_save_path']}")
    else:
        print(f"Warning: Stage 1 text projection head weights not found: {config['text_projection_head_stage1_save_path']}. Using default initialization.")
    sat_projection_head.eval()
    text_projection_head.eval()
    for param in sat_projection_head.parameters():
        param.requires_grad = False
    for param in text_projection_head.parameters():
        param.requires_grad = False


    text_encoder_for_generator_input = None
    text_embedding_dim = 0
    if 'text_encoder_model_name' in config:
        temp_text_encoder = OpenCLIPTextEncoder(
            model_name=config['text_encoder_model_name'],
            pretrained=config['text_encoder_pretrained']
        ).to(device)
        if os.path.exists(config['text_encoder_stage1_save_path']):
            temp_text_encoder.load_state_dict(torch.load(config['text_encoder_stage1_save_path'], map_location=device))
            print(f"Loaded Stage 1 trained text encoder weights: {config['text_encoder_stage1_save_path']}")
        else:
            print(f"Warning: Stage 1 text encoder weights not found: {config['text_encoder_stage1_save_path']}. Using default initialization.")
        temp_text_encoder.eval()
        for param in temp_text_encoder.parameters():
            param.requires_grad = False
        text_encoder_for_generator_input = temp_text_encoder

        import open_clip
        dummy_tokens = open_clip.tokenize(["dummy"])
        max_length = dummy_tokens.shape[1]
        dummy_text_input = open_clip.tokenize(["dummy"])
        dummy_text_input = dummy_text_input.to(device)
        text_embedding_dim = temp_text_encoder(dummy_text_input).shape[-1]
        print(f"Text encoder original output dimension (for projection): {text_embedding_dim}")

    generator_input_dim = sat_projection_head.proj[-1].out_features + text_projection_head.proj[-1].out_features
    generator = Generator_MLP(generator_input_dim, openclip_native_dim, config['g_hidden_dim'], config['g_num_layers']).to(device)
    discriminator = Discriminator_MLP(openclip_native_dim, config['d_hidden_dim'], config['d_num_layers']).to(device)


    target_street_extractor_frozen = OpenCLIPVisionEncoder(
        model_name=config['vision_encoder_model_name'],
        pretrained=config['vision_encoder_pretrained'],
        unfreeze_last_n_layers=0
    ).to(device)
    target_street_extractor_frozen.eval()
    print(f"All model components initialized.")


    transform_common = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    transform_sat_stage2 = transforms.Compose([
        transforms.Resize((config['image_size'], config['image_size'])),
        transform_common
    ])
    full_df_cvusa = pd.read_csv(config['cvusa_csv_path'], header=None, names=config['csv_file_col_names'])

    train_df, val_df = train_test_split(full_df_cvusa,
                                         test_size=config['validation_split_ratio'],
                                         random_state=config['random_state_split'])
    print(f"Dataset split into training set ({len(train_df)} samples) and validation set ({len(val_df)} samples).")


    train_dataset_stage2 = CVUSACorrespondenceDatasetUpdated(
        df_subset=train_df,
        base_config=config,
        transform_sat=transform_sat_stage2,
        transform_street_crop=transform_common,
        text_json_path=config['cvusa_text_json_path'],
        tokenizer=open_clip.tokenize
    )
    val_dataset_stage2 = CVUSACorrespondenceDatasetUpdated(
        df_subset=val_df,
        base_config=config,
        transform_sat=transform_sat_stage2,
        transform_street_crop=transform_common,
        text_json_path=config['cvusa_text_json_path'],
        tokenizer=open_clip.tokenize
    ) if len(val_df) > 0 else None
    
    train_loader_stage2 = DataLoader(train_dataset_stage2, batch_size=config['batch_size'], shuffle=True, num_workers=4, pin_memory=True, collate_fn=collate_fn_skip_none)
    val_loader_stage2 = DataLoader(val_dataset_stage2, batch_size=config['batch_size'], shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate_fn_skip_none) if val_dataset_stage2 else None


    g_params = list(generator.parameters())
    
    optimizerG = optim.Adam(g_params, lr=config['lr_g_sat_encoder'], betas=(config['beta1_adam'], config['beta2_adam']))
    optimizerD = optim.Adam(discriminator.parameters(), lr=config['lr_d'], betas=(config['beta1_adam'], config['beta2_adam']))
    criterion_GAN = nn.MSELoss()

    best_val_g_loss = float('inf')

    print(f"Starting Stage 2: R3GAN-style generator training ({config['num_epochs_stage2_r3gan']} Epochs)...")
    for epoch in range(config['num_epochs_stage2_r3gan']):
        sat_encoder.eval()
        if text_encoder_for_generator_input:
            text_encoder_for_generator_input.eval()
        generator.train()
        discriminator.train()

        for i, data_batch in enumerate(train_loader_stage2):
            if data_batch is None: continue
            sat_imgs, street_img_crops_stack, sat_texts_tokens = data_batch
            if sat_imgs is None or street_img_crops_stack is None or sat_texts_tokens is None: continue
            
            sat_imgs = sat_imgs.to(device)
            street_img_crops_stack = street_img_crops_stack.to(device)
            sat_texts_tokens = sat_texts_tokens.to(device)
            batch_size = sat_imgs.size(0)


            with torch.no_grad():
                street_crops_reshaped = street_img_crops_stack.view(-1, *street_img_crops_stack.shape[2:])
                street_features_all_crops = target_street_extractor_frozen(street_crops_reshaped)
                street_features_all_crops = street_features_all_crops.view(batch_size, config['num_street_crops'], -1)
                feat_B_real = street_features_all_crops.mean(dim=1)
            

            feat_B_real.requires_grad = True

            with torch.no_grad():
                intermediate_sat_feat = sat_encoder(sat_imgs)
                sat_text_embeddings = text_encoder_for_generator_input(sat_texts_tokens)
            sat_feat_proj = sat_projection_head(intermediate_sat_feat)
            text_feat_proj = text_projection_head(sat_text_embeddings)
            fused_generator_input = torch.cat([sat_feat_proj, text_feat_proj], dim=1)
            
            feat_B_fake = generator(fused_generator_input)

            loss_feature_matching = F.l1_loss(feat_B_fake, feat_B_real)

            optimizerD.zero_grad()
            d_real_out = discriminator(feat_B_real)
            d_fake_out = discriminator(feat_B_fake.detach())

            errD_real = criterion_GAN(d_real_out - d_fake_out.mean(), torch.ones_like(d_real_out))
            errD_fake = criterion_GAN(d_fake_out - d_real_out.mean(), -torch.ones_like(d_fake_out))
            errD = (errD_real + errD_fake) / 2

            r1_penalty = r1_gradient_penalty(d_real_out, feat_B_real)
            errD_total = errD + r1_penalty * (config['lambda_r1'] / 2)
            
            errD_total.backward()
            optimizerD.step()


            optimizerG.zero_grad()

            d_real_out_for_g = discriminator(feat_B_real.detach())
            d_fake_out_for_g = discriminator(feat_B_fake)


            errG_fake = criterion_GAN(d_fake_out_for_g - d_real_out_for_g.mean(), torch.ones_like(d_fake_out_for_g))
            errG_real = criterion_GAN(d_real_out_for_g - d_fake_out_for_g.mean(), -torch.ones_like(d_real_out_for_g))
            errG_gan = (errG_real + errG_fake) / 2

            errG_total = errG_gan + loss_feature_matching * config['lambda_FM']

            errG_total.backward()
            optimizerG.step()

            if i % 100 == 0:
                print(f'[Epoch {epoch+1}/{config["num_epochs_stage2_r3gan"]}][{i}/{len(train_loader_stage2)}] '
                      f'Loss_D: {errD.item():.4f} Loss_G_GAN: {errG_gan.item():.4f} '
                      f'Loss_G_FM: {loss_feature_matching.item():.4f} R1: {r1_penalty.item():.4f}')

        if val_loader_stage2:
            generator.eval()
            discriminator.eval()
            sat_encoder.eval()
            if text_encoder_for_generator_input:
                text_encoder_for_generator_input.eval()

            current_val_g_loss = 0.0
            num_val_samples = 0
            with torch.no_grad():
                for val_batch in val_loader_stage2:
                    if val_batch is None: continue
                    sat_imgs_val, street_img_crops_stack_val, sat_texts_tokens_val = val_batch
                    if sat_imgs_val is None or street_img_crops_stack_val is None or sat_texts_tokens_val is None: continue
                    
                    sat_imgs_val = sat_imgs_val.to(device)
                    street_crops_reshaped_val = street_img_crops_stack_val.view(-1, *street_img_crops_stack_val.shape[2:]).to(device)
                    sat_texts_tokens_val = sat_texts_tokens_val.to(device)

                    feat_B_real_val = target_street_extractor_frozen(street_crops_reshaped_val)
                    feat_B_real_val = feat_B_real_val.view(sat_imgs_val.size(0), config['num_street_crops'], -1).mean(dim=1)
                    
                    intermediate_sat_feat_val = sat_encoder(sat_imgs_val)
                    sat_text_embeddings_val = text_encoder_for_generator_input(sat_texts_tokens_val)
                    
                    sat_feat_proj_val = sat_projection_head(intermediate_sat_feat_val)
                    text_feat_proj_val = text_projection_head(sat_text_embeddings_val)
                    fused_generator_input_val = torch.cat([sat_feat_proj_val, text_feat_proj_val], dim=1)
                    
                    feat_B_fake_val = generator(fused_generator_input_val)
                    
                    d_real_val = discriminator(feat_B_real_val)
                    d_fake_val = discriminator(feat_B_fake_val)

                    
                    val_g_loss_fake = criterion_GAN(d_fake_val - d_real_val.mean(), torch.ones_like(d_fake_val))
                    val_g_loss_real = criterion_GAN(d_real_val - d_fake_val.mean(), -torch.ones_like(d_real_val))
                    val_g_loss_gan = (val_g_loss_real + val_g_loss_fake) / 2
                    
                    
                    val_loss_feature_matching = F.l1_loss(feat_B_fake_val, feat_B_real_val)
                    total_val_loss = val_g_loss_gan + val_loss_feature_matching * config['lambda_FM']
                    
                    current_val_g_loss += total_val_loss.item() * sat_imgs_val.size(0)
                    num_val_samples += sat_imgs_val.size(0)
            
            if num_val_samples > 0:
                avg_val_g_loss = current_val_g_loss / num_val_samples
                print(f"Epoch [{epoch+1}] VAL Generator Total Loss: {avg_val_g_loss:.6f}")

                if avg_val_g_loss < best_val_g_loss:
                    best_val_g_loss = avg_val_g_loss
                    torch.save(sat_encoder.state_dict(), config['overall_best_sat_encoder_save_path'])
                    torch.save(generator.state_dict(), config['overall_best_generator_save_path'])
                    print(f"  ==> New lowest validation G Loss: {best_val_g_loss:.6f}. Best models saved.")

    print("\nR3GAN-style training completed!")
    print(f"Final best validation G loss: {best_val_g_loss:.6f}")



def main_two_stage_pretraining():
    config = CONFIG_R3GAN_STYLE
    

    common_paths_to_check = [
        config['cvusa_text_json_path'],
    ]

    if not os.path.isfile(config['cvusa_text_json_path']):
        print(f"Error: Text JSON file '{config['cvusa_text_json_path']}' not found.")
        return


    train_satellite_text_alignment()


    if not os.path.exists(config['sat_encoder_stage1_save_path']):
        print(f"Error: Stage 1 satellite encoder weights '{config['sat_encoder_stage1_save_path']}' not found. Please run Stage 1 first.")
        return
    if not os.path.exists(config['text_encoder_stage1_save_path']):
        print(f"Error: Stage 1 text encoder weights '{config['text_encoder_stage1_save_path']}' not found. Please run Stage 1 first.")
        return

    if not os.path.exists(config['sat_projection_head_stage1_save_path']):
        print(f"Error: Stage 1 satellite projection head weights '{config['sat_projection_head_stage1_save_path']}' not found. Please run Stage 1 first.")
        return
    if not os.path.exists(config['text_projection_head_stage1_save_path']):
        print(f"Error: Stage 1 text projection head weights '{config['text_projection_head_stage1_save_path']}' not found. Please run Stage 1 first.")
        return
        
    train_r3gan_style_stage2()


if __name__ == '__main__':
    main_two_stage_pretraining()