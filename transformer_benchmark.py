import torch
import torch.nn as nn
import time
from transformer import TimeSeriesTransformer

def benchmark_transformer(device_name, batch_size=32):
    """Бенчмарк трансформера на разных устройствах"""
    device = torch.device(device_name)
    print(f"\n🔥 Тестируем трансформер на {device_name.upper()}...")
    print(f"   Batch size: {batch_size}")
    
    # Создаем модель
    model = TimeSeriesTransformer(
        input_dim=1,
        d_model=128,
        n_heads=8,
        n_encoder_layers=3,
        n_decoder_layers=3,
        d_ff=256,
        max_seq_length=200,
        dropout=0.15
    ).to(device)
    
    # Тестовые данные
    seq_len = 60
    pred_len = 15
    src = torch.randn(batch_size, seq_len, 1).to(device)
    tgt = torch.randn(batch_size, pred_len, 1).to(device)
    
    # Прогрев
    model.train()
    for _ in range(5):
        tgt_mask = model.generate_square_subsequent_mask(pred_len).to(device)
        _ = model(src, tgt, tgt_mask=tgt_mask)
    
    # Бенчмарк
    start_time = time.time()
    iterations = 100
    
    for _ in range(iterations):
        tgt_mask = model.generate_square_subsequent_mask(pred_len).to(device)
        output = model(src, tgt, tgt_mask=tgt_mask)
        loss = output.sum()
        loss.backward()
        model.zero_grad()
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"⏱️  Время выполнения {iterations} итераций: {total_time:.3f} сек")
    print(f"🚀 Скорость: {iterations/total_time:.1f} итераций/сек")
    print(f"📊 Время на итерацию: {total_time/iterations*1000:.1f} мс")
    
    return total_time

def main():
    print("🧪 Бенчмарк трансформера на M1 Mac")
    print("=" * 50)
    
    # Проверяем доступные устройства
    print("📱 Доступные устройства:")
    print(f"   CPU: ✅")
    print(f"   MPS: {'✅' if torch.backends.mps.is_available() else '❌'}")
    
    batch_sizes = [16, 32, 64]
    
    for batch_size in batch_sizes:
        print(f"\n{'='*20} BATCH SIZE {batch_size} {'='*20}")
        
        # Тестируем CPU
        cpu_time = benchmark_transformer('cpu', batch_size)
        
        # Тестируем MPS если доступен
        if torch.backends.mps.is_available():
            mps_time = benchmark_transformer('mps', batch_size)
            
            speedup = cpu_time / mps_time
            print(f"\n🏆 РЕЗУЛЬТАТЫ для batch_size={batch_size}:")
            print(f"   CPU время: {cpu_time:.3f} сек")
            print(f"   MPS время: {mps_time:.3f} сек")
            if speedup > 1:
                print(f"   🚀 MPS быстрее в {speedup:.1f}x раз!")
            else:
                print(f"   🐌 CPU быстрее в {1/speedup:.1f}x раз")

if __name__ == "__main__":
    main() 