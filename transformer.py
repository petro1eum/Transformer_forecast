import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import math
import os
import time
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Оптимизация для M1 Max
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'


class PositionalEncoding(nn.Module):
    """Позиционное кодирование для трансформера"""
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0), :]


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention механизм"""
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = query.size()
        
        # Линейные преобразования и разделение на головы
        Q = self.W_q(query).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Вычисление attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Применение attention к values
        context = torch.matmul(attention_weights, V)
        
        # Объединение голов
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        output = self.W_o(context)
        return output


class FeedForward(nn.Module):
    """Feed Forward сеть"""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(self.relu(self.linear1(x))))


class EncoderLayer(nn.Module):
    """Слой энкодера трансформера"""
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention с residual connection и layer norm
        attn_output = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed forward с residual connection и layer norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class DecoderLayer(nn.Module):
    """Слой декодера трансформера"""
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, enc_output: torch.Tensor, 
                src_mask: Optional[torch.Tensor] = None,
                tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Masked self-attention
        attn_output = self.self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Cross-attention
        attn_output = self.cross_attention(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        
        # Feed forward
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x


class TimeSeriesTransformer(nn.Module):
    """Улучшенная модель трансформера для прогнозирования временных рядов"""
    def __init__(self, 
                 input_dim: int,
                 d_model: int = 512,
                 n_heads: int = 8,
                 n_encoder_layers: int = 6,
                 n_decoder_layers: int = 6,
                 d_ff: int = 2048,
                 max_seq_length: int = 1000,
                 dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        self.input_embedding = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        
        # Learnable time embeddings для лучшего понимания временных паттернов
        self.time_embedding = nn.Parameter(torch.randn(1, max_seq_length, d_model) * 0.02)
        
        # Дополнительные эмбеддинги для лучшего представления
        self.value_embedding = nn.Linear(input_dim, d_model // 4)
        self.trend_embedding = nn.Linear(input_dim, d_model // 4)
        self.seasonal_embedding = nn.Linear(input_dim, d_model // 4)
        self.residual_embedding = nn.Linear(input_dim, d_model // 4)
        
        # Encoder
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_encoder_layers)
        ])
        
        # Decoder
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_decoder_layers)
        ])
        
        # Финальная нормализация перед выходом
        self.final_norm = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, input_dim)
        self.dropout = nn.Dropout(dropout)
        
    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """Создание маски для автогенерации"""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def forward(self, src: torch.Tensor, tgt: torch.Tensor, 
                src_mask: Optional[torch.Tensor] = None,
                tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Улучшенное embedding с декомпозицией
        batch_size, seq_len, _ = src.shape
        
        # Основное embedding
        src_emb = self.input_embedding(src) * math.sqrt(self.d_model)
        
        # Декомпозиция временного ряда на компоненты
        value_emb = self.value_embedding(src)
        trend_emb = self.trend_embedding(src)
        seasonal_emb = self.seasonal_embedding(src)
        residual_emb = self.residual_embedding(src)
        
        # Объединяем все компоненты
        decomposed_emb = torch.cat([value_emb, trend_emb, seasonal_emb, residual_emb], dim=-1)
        src = src_emb + decomposed_emb
        
        src = self.positional_encoding(src)
        
        # Добавляем learnable time embeddings
        if seq_len <= self.max_seq_length:
            src = src + self.time_embedding[:, :seq_len, :]
        
        src = self.dropout(src)
        
        # Прохождение через encoder
        enc_output = src
        for encoder_layer in self.encoder_layers:
            enc_output = encoder_layer(enc_output, src_mask)
        
        # Embedding и позиционное кодирование для целевой последовательности
        tgt = self.input_embedding(tgt) * math.sqrt(self.d_model)
        tgt = self.positional_encoding(tgt)
        
        # Добавляем learnable time embeddings для target
        tgt_len = tgt.size(1)
        if tgt_len <= self.max_seq_length:
            tgt = tgt + self.time_embedding[:, :tgt_len, :]
        
        tgt = self.dropout(tgt)
        
        # Прохождение через decoder
        dec_output = tgt
        for decoder_layer in self.decoder_layers:
            dec_output = decoder_layer(dec_output, enc_output, src_mask, tgt_mask)
        
        # Финальная нормализация и проекция
        dec_output = self.final_norm(dec_output)
        output = self.output_projection(dec_output)
        
        return output


class TimeSeriesDataset(Dataset):
    """Dataset для временных рядов с аугментацией"""
    def __init__(self, data: np.ndarray, seq_length: int, pred_length: int, augment: bool = True):
        self.data = torch.FloatTensor(data)
        self.seq_length = seq_length
        self.pred_length = pred_length
        self.augment = augment
        
    def __len__(self):
        return len(self.data) - self.seq_length - self.pred_length + 1
    
    def __getitem__(self, idx):
        src = self.data[idx:idx + self.seq_length].clone()
        tgt = self.data[idx + self.seq_length:idx + self.seq_length + self.pred_length].clone()
        
        # Аугментация данных для обучения
        if self.augment and torch.rand(1) < 0.3:  # 30% вероятность аугментации
            # Добавляем небольшой шум
            noise_std = 0.02 * src.std()
            src += torch.randn_like(src) * noise_std
            tgt += torch.randn_like(tgt) * noise_std
            
            # Масштабирование
            if torch.rand(1) < 0.5:
                scale = 0.95 + torch.rand(1) * 0.1  # 0.95-1.05
                src *= scale
                tgt *= scale
        
        return src, tgt


def create_synthetic_data(n_samples: int = 1000) -> np.ndarray:
    """Создание синтетических данных для демонстрации"""
    np.random.seed(42)  # Фиксируем seed для воспроизводимости
    time = np.arange(n_samples)
    # Комбинация тренда, сезонности и шума
    trend = 0.02 * time  # Уменьшаем тренд
    seasonal1 = 8 * np.sin(2 * np.pi * time / 50)  # Основная сезонность
    seasonal2 = 3 * np.sin(2 * np.pi * time / 100)  # Долгосрочная сезонность
    seasonal3 = 2 * np.sin(2 * np.pi * time / 25)   # Короткосрочная сезонность
    noise = np.random.normal(0, 1.5, n_samples)  # Уменьшаем шум
    data = trend + seasonal1 + seasonal2 + seasonal3 + noise + 20  # Добавляем базовый уровень
    return data.reshape(-1, 1)


def trend_aware_loss(predictions: torch.Tensor, targets: torch.Tensor, alpha: float = 0.3) -> torch.Tensor:
    """Функция потерь с учетом тренда"""
    mse_loss = nn.MSELoss()(predictions, targets)
    
    # Потери на производной (тренд)
    pred_diff = predictions[:, 1:] - predictions[:, :-1]
    target_diff = targets[:, 1:] - targets[:, :-1]
    trend_loss = nn.MSELoss()(pred_diff, target_diff)
    
    return mse_loss + alpha * trend_loss

def train_model(model: nn.Module, 
                train_loader: DataLoader, 
                val_loader: DataLoader,
                n_epochs: int = 100,
                learning_rate: float = 0.001,
                device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
    """Обучение модели с подробным логированием времени и early stopping"""
    model = model.to(device)
    criterion = nn.MSELoss()  # Возвращаемся к простой MSE
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)  # L2 регуляризация
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    train_losses = []
    val_losses = []
    
    # Early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 10
    best_model_state = None
    
    print(f"\n🚀 Начинаем обучение на {len(train_loader)} батчах...")
    print(f"   🛑 Early stopping: patience={patience}")
    training_start_time = time.time()
    
    for epoch in range(n_epochs):
        epoch_start_time = time.time()
        # Обучение
        model.train()
        train_loss = 0
        train_start_time = time.time()
        
        for batch_idx, (src, tgt) in enumerate(train_loader):
            src, tgt = src.to(device), tgt.to(device)
            
            # Создание входа для декодера (сдвинутая целевая последовательность)
            tgt_input = torch.cat([src[:, -1:, :], tgt[:, :-1, :]], dim=1)
            
            # Создание маски для декодера
            tgt_mask = model.generate_square_subsequent_mask(tgt_input.size(1)).to(device)
            
            # Forward pass
            optimizer.zero_grad()
            output = model(src, tgt_input, tgt_mask=tgt_mask)
            loss = criterion(output, tgt)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        train_time = time.time() - train_start_time
        
        # Валидация
        model.eval()
        val_loss = 0
        val_start_time = time.time()
        
        with torch.no_grad():
            for src, tgt in val_loader:
                src, tgt = src.to(device), tgt.to(device)
                tgt_input = torch.cat([src[:, -1:, :], tgt[:, :-1, :]], dim=1)
                tgt_mask = model.generate_square_subsequent_mask(tgt_input.size(1)).to(device)
                output = model(src, tgt_input, tgt_mask=tgt_mask)
                loss = criterion(output, tgt)
                val_loss += loss.item()
        
        val_time = time.time() - val_start_time
        epoch_time = time.time() - epoch_start_time
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        scheduler.step(avg_val_loss)
        
        # Early stopping logic
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        if epoch % 5 == 0:  # Уведомления каждые 5 эпох вместо 10
            print(f'Epoch {epoch:3d}/{n_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {optimizer.param_groups[0]["lr"]:.6f}')
            print(f'   ⏱️  Train: {train_time:.2f}s | Val: {val_time:.2f}s | Total: {epoch_time:.2f}s | Speed: {len(train_loader)/train_time:.1f} batch/s')
            print(f'   🎯 Best Val Loss: {best_val_loss:.4f} | Patience: {patience_counter}/{patience}')
        
        # Early stopping check
        if patience_counter >= patience:
            print(f'\n🛑 Early stopping на эпохе {epoch}! Val loss не улучшался {patience} эпох.')
            break
    
    # Загружаем лучшую модель
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"✅ Загружена лучшая модель с Val Loss: {best_val_loss:.6f}")
    
    total_training_time = time.time() - training_start_time
    print(f"\n🎯 Обучение завершено!")
    print(f"   Общее время: {total_training_time:.1f}s ({total_training_time/60:.1f} мин)")
    print(f"   Время на эпоху: {total_training_time/len(train_losses):.1f}s")
    print(f"   Финальный Train Loss: {train_losses[-1]:.6f}")
    print(f"   Лучший Val Loss: {best_val_loss:.6f}")
    
    return train_losses, val_losses


def forecast(model: nn.Module, 
             src_sequence: torch.Tensor, 
             n_steps: int,
             device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
             temperature: float = 1.0) -> torch.Tensor:
    """Улучшенное прогнозирование с температурой и ансамблем"""
    model.eval()
    src_sequence = src_sequence.to(device)
    
    predictions = []
    
    # Ансамбль из нескольких прогнозов с разным dropout
    for ensemble_idx in range(3):
        model.train() if ensemble_idx > 0 else model.eval()  # Включаем dropout для разнообразия
        
        with torch.no_grad():
            # Начальная последовательность для декодера
            dec_input = src_sequence[:, -1:, :]
            
            for step in range(n_steps):
                tgt_mask = model.generate_square_subsequent_mask(dec_input.size(1)).to(device)
                output = model(src_sequence, dec_input, tgt_mask=tgt_mask)
                next_pred = output[:, -1:, :]
                
                # Применяем температуру для разнообразия
                if temperature != 1.0 and ensemble_idx > 0:
                    next_pred = next_pred / temperature
                
                dec_input = torch.cat([dec_input, next_pred], dim=1)
        
        predictions.append(dec_input[:, 1:, :])
    
    # Усредняем предсказания ансамбля
    ensemble_pred = torch.stack(predictions).mean(dim=0)
    return ensemble_pred


def visualize_results(train_data: np.ndarray, 
                     test_data: np.ndarray,
                     predictions: np.ndarray,
                     train_losses: list,
                     val_losses: list):
    """Визуализация результатов"""
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # График прогнозов
    axes[0].plot(range(len(train_data)), train_data, label='Train Data', alpha=0.7)
    axes[0].plot(range(len(train_data), len(train_data) + len(test_data)), 
                test_data, label='Actual Test Data', alpha=0.7)
    axes[0].plot(range(len(train_data), len(train_data) + len(predictions)), 
                predictions, label='Predictions', linestyle='--', alpha=0.8)
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Value')
    axes[0].set_title('Time Series Forecasting with Transformer')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # График функции потерь
    axes[1].plot(train_losses, label='Train Loss')
    axes[1].plot(val_losses, label='Validation Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MSE Loss')
    axes[1].set_title('Training Progress')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def main():
    """Основная функция для запуска примера"""
    script_start_time = time.time()
    
    # Параметры
    seq_length = 60      # Увеличиваем окно наблюдения
    pred_length = 15     # Уменьшаем горизонт прогноза
    batch_size = 32      # Уменьшаем для лучшей генерализации
    n_epochs = 80        # Больше эпох для лучшего обучения
    
    # Создание данных
    print("📊 Создание синтетических данных...")
    data_start_time = time.time()
    data = create_synthetic_data(2000)
    print(f"   ✅ Данные созданы за {time.time() - data_start_time:.2f}s")
    
    # Разделение на обучающую и тестовую выборки
    train_size = int(0.8 * len(data))
    train_data = data[:train_size]
    test_data = data[train_size:]
    
    # Нормализация данных
    mean = train_data.mean()
    std = train_data.std()
    train_data_norm = (train_data - mean) / std
    test_data_norm = (test_data - mean) / std
    
    # Создание датасетов с аугментацией
    train_dataset = TimeSeriesDataset(train_data_norm[:-100], seq_length, pred_length, augment=False)
    val_dataset = TimeSeriesDataset(train_data_norm[-100-seq_length-pred_length:], seq_length, pred_length, augment=False)
    
    # Оптимизированные DataLoader для M1 Max (10 ядер)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=6, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                           num_workers=4, pin_memory=True, persistent_workers=True)
    
    # Определяем устройство (оптимизация для M1/M2 Mac)
    if torch.backends.mps.is_available():
        device = 'mps'
        print("🚀 Используем Metal Performance Shaders (MPS) для ускорения!")
        print(f"   MPS built: {torch.backends.mps.is_built()}")
        # Очищаем кэш MPS для оптимальной производительности
        if hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()
    elif torch.cuda.is_available():
        device = 'cuda'
        print("Используем CUDA для ускорения!")
    else:
        device = 'cpu'
        print("Используем CPU (медленно)")
    
    # Создание модели
    print("\n🧠 Создание модели трансформера...")
    model_start_time = time.time()
    model = TimeSeriesTransformer(
        input_dim=1,
        d_model=128,     # Оптимальный размер
        n_heads=8,       # 8 голов внимания
        n_encoder_layers=4,  # Немного больше для лучшего понимания
        n_decoder_layers=4,  # Симметрично
        d_ff=512,        # Увеличиваем feed-forward для лучшей экспрессивности
        max_seq_length=200,
        dropout=0.2      # Умеренный dropout
    )
    
    model_creation_time = time.time() - model_start_time
    print(f"   ✅ Модель создана за {model_creation_time:.2f}s")
    print(f"   📊 Количество параметров: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   🖥️  Устройство: {device}")
    print(f"   📈 Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")
    print(f"   ⚙️  Batch size: {batch_size} | Epochs: {n_epochs}")
    
    # Обучение модели
    print("Обучение модели...")
    train_losses, val_losses = train_model(model, train_loader, val_loader, n_epochs=n_epochs, learning_rate=0.0005, device=device)
    
    # Прогнозирование на тестовых данных
    print("\n🔮 Выполнение прогнозирования...")
    print(f"   📏 Длина тестовых данных: {len(test_data_norm)}")
    prediction_start_time = time.time()
    model.eval()
    test_predictions = []
    
    # Используем скользящее окно для прогнозирования
    prediction_steps = 0
    for i in range(0, len(test_data_norm) - seq_length, pred_length):
        src = torch.FloatTensor(test_data_norm[i:i+seq_length]).unsqueeze(0)
        pred = forecast(model, src, pred_length, device, temperature=1.0)  # Стандартная температура
        test_predictions.extend(pred.squeeze().cpu().numpy())
        prediction_steps += 1
        
        if prediction_steps % 5 == 0:
            print(f"   🔄 Выполнено {prediction_steps} шагов прогнозирования...")
        
        if i + seq_length + pred_length >= len(test_data_norm):
            break
    
    prediction_time = time.time() - prediction_start_time
    print(f"   ✅ Прогнозирование завершено за {prediction_time:.2f}s")
    print(f"   📊 Всего шагов: {prediction_steps} | Предсказаний: {len(test_predictions)}")
    print(f"   ⚡ Скорость: {prediction_steps/prediction_time:.1f} шагов/сек")
    
    # Денормализация предсказаний
    test_predictions = np.array(test_predictions) * std + mean
    
    # Визуализация результатов
    print("\n📊 Визуализация результатов...")
    viz_start_time = time.time()
    visualize_results(
        train_data.squeeze(), 
        test_data[:len(test_predictions)].squeeze(),
        test_predictions.squeeze(),
        train_losses,
        val_losses
    )
    viz_time = time.time() - viz_start_time
    print(f"   ✅ Графики построены за {viz_time:.2f}s")
    
    # Расчет метрик
    actual = test_data[:len(test_predictions)].squeeze()
    predicted = test_predictions.squeeze()
    mse = np.mean((actual - predicted) ** 2)
    mae = np.mean(np.abs(actual - predicted))
    
    print(f"\n📈 Метрики на тестовых данных:")
    print(f"   MSE: {mse:.4f}")
    print(f"   MAE: {mae:.4f}")
    print(f"   RMSE: {np.sqrt(mse):.4f}")
    
    # Сохранение модели
    save_start_time = time.time()
    torch.save({
        'model_state_dict': model.state_dict(),
        'mean': mean,
        'std': std,
        'model_config': {
            'input_dim': 1,
            'd_model': 128,
            'n_heads': 8,
            'n_encoder_layers': 3,
            'n_decoder_layers': 3,
            'd_ff': 256,
            'max_seq_length': 200,
            'dropout': 0.25
        }
    }, 'transformer_timeseries_model.pth')
    save_time = time.time() - save_start_time
    print(f"\n💾 Модель сохранена за {save_time:.2f}s в 'transformer_timeseries_model.pth'")
    
    # Общее время выполнения
    total_script_time = time.time() - script_start_time
    print(f"\n🎯 ОБЩЕЕ ВРЕМЯ ВЫПОЛНЕНИЯ: {total_script_time:.1f}s ({total_script_time/60:.1f} мин)")
    print(f"   📊 Подготовка данных: {data_start_time:.2f}s")
    print(f"   🧠 Создание модели: {model_creation_time:.2f}s") 
    print(f"   🚀 Обучение: {total_script_time - prediction_start_time + prediction_time:.1f}s")
    print(f"   🔮 Прогнозирование: {prediction_time:.2f}s")
    print(f"   📊 Визуализация: {viz_time:.2f}s")


if __name__ == "__main__":
    main()