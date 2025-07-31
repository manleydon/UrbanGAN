import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import KBinsDiscretizer
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import os

from collections import OrderedDict
import open_clip
import json
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

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

CONFIG_DOWNSTREAM = {
    "population_train_val_csv_path": "./data/Haidian_train.csv",
    "test_csv_path": "./data/Haidian_test.csv",
    "csv_filename_column": "Filename",
    "image_path_prefix_for_pop_data": "./data/Haidian/images/",
    "csv_path_is_absolute_for_pop_data": False,
    "image_size_sat": 224,
    "vision_encoder_model_name": "ViT-B-16",
    "vision_encoder_pretrained": "laion2b_s34b_b88k",
    "vision_encoder_unfreeze_layers": 0,

    "sat_encoder_stage1_load_path": "sat_encoder_stage1_aligned_freeze.pth",
    "generator_load_path": "r3gan_style_generator_best_freeze.pth",
    "g_hidden_dim": 1024,
    "g_num_layers": 4,
    "xgboost_params": {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'eta': 0.05,
        'max_depth': 7,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 1,
        'gamma': 0,
        'lambda': 1,
        'alpha': 0,
    },
    "xgboost_n_estimators": 1000,
    "xgboost_early_stopping_rounds": 50,
    "train_val_split_ratio": 0.2,
    "random_state_split": 42,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "model_save_path_C": "predictor_C_DOWNSTREAM_combined.ubj",
    "output_eval_predictions_csv_path": "DOWNSTREAM_ablation_TEST_predictions_freeze.csv",
    "output_eval_metrics_csv_path": "DOWNSTREAM_eval_freeze.csv",
    "sat_projection_head_stage1_path": "sat_projection_head_stage1_freeze.pth",
    "text_projection_head_stage1_path": "text_projection_head_stage1_freeze.pth",
    "sat_patch_projection_head_stage1_path": "sat_patch_projection_head_stage1.pth",
}

CONFIG_DOWNSTREAM.update({
    "caption_json_path": "./data/Haidiancaptions_output.json",
    "text_encoder_stage1_load_path": "text_encoder_stage1_aligned_freeze.pth",
    "text_encoder_model_name": "ViT-B-16",
    "text_encoder_pretrained": "laion2b_s34b_b88k",
    "context_length": 77,
})

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

with open(CONFIG_DOWNSTREAM['caption_json_path'], 'r', encoding='utf-8') as f:
    caption_json = json.load(f)
img2caption = {}
for item_list in caption_json:
    if not item_list: continue
    img_path = os.path.basename(item_list[0]['image'])
    captions = [x['caption'] for x in item_list]
    img2caption[img_path] = captions

class FeatureExtractionDataset(Dataset):
    def __init__(self, df, config, img2caption, transform, label_column=None):
        self.df = df.reset_index(drop=True)
        self.config = config
        self.img2caption = img2caption
        self.transform = transform
        self.label_column = label_column

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filename = str(row[self.config['csv_filename_column']]).strip()
        if not self.config['csv_path_is_absolute_for_pop_data']:
            filename = filename.lstrip('/\\')
            image_full_path = os.path.join(self.config['image_path_prefix_for_pop_data'], filename)
        else:
            image_full_path = filename
        try:
            with Image.open(image_full_path) as img:
                img_rgb = img.convert('RGB')
                img_tensor = self.transform(img_rgb)
        except Exception as e:
            return None
        img_key = os.path.basename(filename)
        captions = self.img2caption.get(img_key, None)
        if not captions:
            return None
        caption = np.random.choice(captions)
        text_tokens = open_clip.tokenize([caption], context_length=self.config.get('context_length', 77)).squeeze(0)
        label = 0.0
        if self.label_column is not None:
            label = float(row[self.label_column])
        return img_tensor, text_tokens, label

def collate_fn_filter_none(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    imgs, text_tokens, labels = zip(*batch)
    return torch.stack(imgs), torch.stack(text_tokens), torch.tensor(labels)


def get_features_C_branch(df, config, sat_encoder_model, gan_generator_model, text_encoder_model, img2caption, label_column=None, batch_size=32, num_workers=4):
    device = config['device']
    transform_sat = transforms.Compose([
        transforms.Resize((config['image_size_sat'], config['image_size_sat'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = FeatureExtractionDataset(df, config, img2caption, transform_sat, label_column)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=False, collate_fn=collate_fn_filter_none)
    features_list = []
    labels_list = []
    sat_encoder_model.eval()
    gan_generator_model.eval()
    text_encoder_model.eval()

    temp_openclip_encoder = OpenCLIPVisionEncoder(
        model_name=config['vision_encoder_model_name'],
        pretrained=config['vision_encoder_pretrained']
    )
    openclip_dim = temp_openclip_encoder.output_dim
    del temp_openclip_encoder
    
    sat_projection_head = ProjectionHead(512, 768).to(device)
    text_projection_head = ProjectionHead(512, 768).to(device)
    sat_projection_head.load_state_dict(torch.load(config['sat_projection_head_stage1_path'], map_location=device))
    text_projection_head.load_state_dict(torch.load(config['text_projection_head_stage1_path'], map_location=device))
    sat_projection_head.eval()
    text_projection_head.eval()
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Feature extraction', mininterval=1.0):
            if batch is None:
                continue
            imgs, text_tokens, labels = batch
            if imgs is None or text_tokens is None: 
                continue

            imgs = imgs.to(device)
            text_tokens = text_tokens.to(device)

            feat_A_tensor_orig = sat_encoder_model(imgs)
            text_feat_orig = text_encoder_model(text_tokens)
            
            feat_A_proj = sat_projection_head(feat_A_tensor_orig)
            text_feat_proj = text_projection_head(text_feat_orig)
            
            fused_input = torch.cat([feat_A_proj, text_feat_proj], dim=1)
            
            feat_B_tensor = gan_generator_model(fused_input)
            
            feat_A_orig_np = feat_A_tensor_orig.cpu().numpy()
            feat_A_proj_np = feat_A_proj.cpu().numpy()
            text_feat_orig_np = text_feat_orig.cpu().numpy()
            text_feat_proj_np = text_feat_proj.cpu().numpy()
            feat_B_np = feat_B_tensor.cpu().numpy()
            
            current_feature = np.concatenate((
                feat_A_orig_np,
                feat_A_proj_np,
                feat_B_np
            ), axis=1)
            
            features_list.append(current_feature)
            if label_column is not None:
                labels_list.append(labels.numpy())
                
    features = np.concatenate(features_list, axis=0)
    labels = np.concatenate(labels_list, axis=0) if label_column is not None else np.array([])
    
    print(f"Feature dimension description:")
    print(f"- Original satellite features: {feat_A_orig_np.shape[1]} dims")
    print(f"- Projected satellite features: {feat_A_proj_np.shape[1]} dims")
    print(f"- Original text features: {text_feat_orig_np.shape[1]} dims")
    print(f"- Projected text features: {text_feat_proj_np.shape[1]} dims")
    print(f"- Generator output features: {feat_B_np.shape[1]} dims")
    print(f"- Total feature dimensions: {features.shape[1]} dims")
    
    return features, labels

def train_xgboost_model(X_train, y_train, X_val, y_val, config, model_save_path=None):
    if X_train.ndim == 1 and X_train.shape[0] > 0 : X_train = X_train.reshape(1, -1)
    if X_val.ndim == 1 and X_val.shape[0] > 0 : X_val = X_val.reshape(1, -1)
    if X_train.shape[0] == 0: print(f"Training set is empty, skipping."); return None
    eval_set_param = None; early_stopping_rounds_param = config['xgboost_early_stopping_rounds']
    if X_val.shape[0] > 0: eval_set_param = [(X_val, y_val)]
    else: print(f"Warning: Validation set is empty. Early stopping disabled."); early_stopping_rounds_param = None
    xgb_reg = xgb.XGBRegressor(**config['xgboost_params'], n_estimators=config['xgboost_n_estimators'],
                               early_stopping_rounds=early_stopping_rounds_param, random_state=config['random_state_split'])
    print(f"Starting XGBoost training (input feature dimension: {X_train.shape[1]})...")
    fit_params = { "X": X_train, "y": y_train, "verbose": 100 }
    if eval_set_param: fit_params["eval_set"] = eval_set_param
    xgb_reg.fit(**fit_params)
    print(f"Model training completed.")
    if X_val.shape[0] > 0:
        y_pred_val = xgb_reg.predict(X_val)
        r2 = r2_score(y_val, y_pred_val); rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
        print(f"  Validation R²: {r2:.4f}, RMSE: {rmse:.4f}")
    if model_save_path:
        xgb_reg.save_model(model_save_path); print(f"  Model saved to: {model_save_path}")
    return xgb_reg

def create_population_strata(y_values, n_bins=10):
    kbd = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
    y_binned = kbd.fit_transform(y_values.reshape(-1, 1))
    return y_binned.flatten().astype(int)

def run_DOWNSTREAM_training():
    config = CONFIG_DOWNSTREAM
    device = config['device']
    print(f"Using device: {device}")
    print("Loading OpenCLIP Vision Encoder (stage1 weights)...")
    sat_encoder = OpenCLIPVisionEncoder(
        model_name=config['vision_encoder_model_name'],
        pretrained=config['vision_encoder_pretrained'],
        unfreeze_last_n_layers=config['vision_encoder_unfreeze_layers']
    ).to(device)
    state_dict_sat = torch.load(config['sat_encoder_stage1_load_path'], map_location=device)
    if any(k.startswith('module.') for k in state_dict_sat.keys()):
        new_state_dict_sat = OrderedDict()
        for k, v in state_dict_sat.items():
            name = k[7:]
            new_state_dict_sat[name] = v
        sat_encoder.load_state_dict(new_state_dict_sat)
    else:
        sat_encoder.load_state_dict(state_dict_sat)
    sat_encoder.eval()
    print("Loading Generator (stage2 weights)...")
    temp_openclip_encoder = OpenCLIPVisionEncoder(
        model_name=config['vision_encoder_model_name'],
        pretrained=config['vision_encoder_pretrained']
    )
    openclip_dim = temp_openclip_encoder.output_dim
    del temp_openclip_encoder
    gan_generator = Generator_MLP(
        input_dim=768 + 768,
        output_dim=openclip_dim,
        hidden_dim=config['g_hidden_dim'],
        num_layers=config['g_num_layers']
    ).to(device)
    state_dict_gen = torch.load(config['generator_load_path'], map_location=device)
    if any(k.startswith('module.') for k in state_dict_gen.keys()):
        new_state_dict_gen = OrderedDict()
        for k, v in state_dict_gen.items():
            name = k[7:]
            new_state_dict_gen[name] = v
        gan_generator.load_state_dict(new_state_dict_gen)
    else:
        gan_generator.load_state_dict(state_dict_gen)
    gan_generator.eval()
    print("Loading text encoder (stage1 weights)...")
    text_encoder = OpenCLIPTextEncoder(
        model_name=config['text_encoder_model_name'],
        pretrained=config['text_encoder_pretrained']
    ).to(device)
    state_dict_text = torch.load(config['text_encoder_stage1_load_path'], map_location=device)
    text_encoder.load_state_dict(state_dict_text)
    text_encoder.eval()
    print(f"Loading training data: {config['population_train_val_csv_path']}")
    df = pd.read_csv(config['population_train_val_csv_path'])

    valid_mask = (
        df['population'].notnull() & (df['population'] != 0) &
        df['log_Carbon'].notnull() & (df['log_Carbon'] != 0) &
        df['BuildingHeight'].notnull() & (df['BuildingHeight'] != 0)
    )
    df_filtered = df[valid_mask]
    if len(df_filtered) == 0:
        print("  Skipping, no valid data for all targets."); return
    y_population = df_filtered['population'].values
    y_log_Carbon = df_filtered['log_Carbon'].values
    y_BuildingHeight = df_filtered['BuildingHeight'].values
    for target_col, y_full in zip(['population', 'log_Carbon', 'BuildingHeight'], [y_population, y_log_Carbon, y_BuildingHeight]):
        print(f"\n=== Current training target: {target_col} ===")
        config['csv_population_column'] = target_col
        X_full, y_full = get_features_C_branch(
            df_filtered, config, sat_encoder, gan_generator, text_encoder, img2caption,
            label_column=target_col, batch_size=32, num_workers=4
        )
        print(f"  Feature extraction completed. Samples: {X_full.shape[0]}, Feature dimensions: {X_full.shape[1]}")
        strata = create_population_strata(y_full, n_bins=10)
        X_train, X_val, y_train, y_val = train_test_split(
            X_full, y_full,
            test_size=config['train_val_split_ratio'],
            random_state=config['random_state_split'],
            stratify=strata
        )
        final_model_save_path = f"{target_col}_" + config["model_save_path_C"]
        print(f"\n--- Training and saving model ---")
        train_xgboost_model(X_train, y_train, X_val, y_val, config, final_model_save_path)
    print("\nModel training completed for all specified targets.")

def run_DOWNSTREAM_evaluation():
    config = CONFIG_DOWNSTREAM
    device = config['device']
    print(f"Using device: {device}")
    print("Loading OpenCLIP Vision Encoder (stage1 weights)...")
    sat_encoder = OpenCLIPVisionEncoder(
        model_name=config['vision_encoder_model_name'],
        pretrained=config['vision_encoder_pretrained'],
        unfreeze_last_n_layers=config['vision_encoder_unfreeze_layers']
    ).to(device)
    state_dict_sat = torch.load(config['sat_encoder_stage1_load_path'], map_location=device)
    if any(k.startswith('module.') for k in state_dict_sat.keys()):
        new_state_dict_sat = OrderedDict()
        for k, v in state_dict_sat.items():
            name = k[7:]
            new_state_dict_sat[name] = v
        sat_encoder.load_state_dict(new_state_dict_sat)
    else:
        sat_encoder.load_state_dict(state_dict_sat)
    sat_encoder.eval()
    print("Loading Generator (stage2 weights)...")
    temp_openclip_encoder = OpenCLIPVisionEncoder(
        model_name=config['vision_encoder_model_name'],
        pretrained=config['vision_encoder_pretrained']
    )
    openclip_dim = temp_openclip_encoder.output_dim
    del temp_openclip_encoder
    gan_generator = Generator_MLP(
        input_dim=768 + 768,
        output_dim=openclip_dim,
        hidden_dim=config['g_hidden_dim'],
        num_layers=config['g_num_layers']
    ).to(device)
    state_dict_gen = torch.load(config['generator_load_path'], map_location=device)
    if any(k.startswith('module.') for k in state_dict_gen.keys()):
        new_state_dict_gen = OrderedDict()
        for k, v in state_dict_gen.items():
            name = k[7:]
            new_state_dict_gen[name] = v
        gan_generator.load_state_dict(new_state_dict_gen)
    else:
        gan_generator.load_state_dict(state_dict_gen)
    gan_generator.eval()
    print("Loading text encoder (stage1 weights)...")
    text_encoder = OpenCLIPTextEncoder(
        model_name=config['text_encoder_model_name'],
        pretrained=config['text_encoder_pretrained']
    ).to(device)
    state_dict_text = torch.load(config['text_encoder_stage1_load_path'], map_location=device)
    text_encoder.load_state_dict(state_dict_text)
    text_encoder.eval()
    print(f"Loading test data: {config['test_csv_path']}")
    df = pd.read_csv(config['test_csv_path'])
    for target_col in ['population', 'log_Carbon', 'BuildingHeight']:
        print(f"\n=== Current evaluation target: {target_col} ===")
        config['csv_population_column'] = target_col
        df_filtered = df[df[target_col].notnull() & (df[target_col] != 0)]
        if len(df_filtered) == 0:
            print(f"  Skipping {target_col}, no valid data."); continue
        X_full, y_full = get_features_C_branch(df_filtered, config, sat_encoder, gan_generator, text_encoder, img2caption, label_column=target_col)
        print(f"  Feature extraction completed. Samples: {X_full.shape[0]}, Feature dimensions: {X_full.shape[1]}")
        model_path = f"{target_col}_" + config["model_save_path_C"]
        xgb_model = xgb.XGBRegressor()
        xgb_model.load_model(model_path)
        y_pred = xgb_model.predict(X_full)
        r2 = r2_score(y_full, y_pred)
        mae = mean_absolute_error(y_full, y_pred)
        rmse = np.sqrt(mean_squared_error(y_full, y_pred))
        print(f"  R²: {r2:.4f}  MAE: {mae:.4f}  RMSE: {rmse:.4f}")

        output_csv = f"{target_col}_" + config['output_eval_predictions_csv_path']
        pd.DataFrame({
            config['csv_filename_column']: df_filtered[config['csv_filename_column']].values,
            target_col: y_full,
            "predicted_C": y_pred
        }).to_csv(output_csv, index=False)
        print(f"  Prediction results saved to: {output_csv}")
        metrics_csv = config['output_eval_metrics_csv_path']
        metrics_df = pd.DataFrame([{
            'Model': 'C',
            'Target': target_col,
            'R2': r2,
            'RMSE': rmse,
            'MAE': mae,
            'Valid_Samples': len(y_full),
            'Total_Samples': len(df_filtered)
        }])
        if not os.path.exists(metrics_csv):
            metrics_df.to_csv(metrics_csv, index=False)
        else:
            existing_df = pd.read_csv(metrics_csv)
            existing_df = existing_df[~((existing_df['Target'] == target_col) & (existing_df['Model'] == 'C'))]
            updated_df = pd.concat([existing_df, metrics_df], ignore_index=True)
            updated_df.to_csv(metrics_csv, index=False)
        print(f"  Evaluation metrics saved to: {metrics_csv}")
    print("\nEvaluation completed for all specified targets.")

if __name__ == '__main__':
    config = CONFIG_DOWNSTREAM
    paths_to_check = [
        config['population_train_val_csv_path'],
        config['test_csv_path'],
        config['sat_encoder_stage1_load_path'],
        config['generator_load_path'],
        config['sat_projection_head_stage1_path'],
        config['text_projection_head_stage1_path'],
        config['sat_patch_projection_head_stage1_path'],
        config['text_encoder_stage1_load_path'],
    ]
    all_exist = True
    for p in paths_to_check:
        if not (os.path.exists(p) and (os.path.isfile(p) or os.path.isdir(p))):
            print(f"Error: Required file or directory '{p}' not found or incorrect type.")
            all_exist = False
    if all_exist:
        run_DOWNSTREAM_training()
        run_DOWNSTREAM_evaluation()
    else:
        print("Required files/directories missing, terminating.")