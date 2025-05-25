import torch
import time
import numpy as np

def benchmark_device(device_name):
    """Бенчмарк для сравнения устройств"""
    device = torch.device(device_name)
    print(f"\n🔥 Тестируем {device_name.upper()}...")
    
    # Создаем тестовые данные
    batch_size = 32
    seq_len = 60
    d_model = 128
    
    # Тестовые тензоры
    x = torch.randn(batch_size, seq_len, d_model).to(device)
    linear = torch.nn.Linear(d_model, d_model).to(device)
    
    # Прогрев
    for _ in range(10):
        _ = linear(x)
    
    # Бенчмарк
    start_time = time.time()
    iterations = 1000
    
    for _ in range(iterations):
        output = linear(x)
        loss = output.sum()
        loss.backward()
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"⏱️  Время выполнения {iterations} итераций: {total_time:.3f} сек")
    print(f"🚀 Скорость: {iterations/total_time:.1f} итераций/сек")
    
    return total_time

def main():
    print("🧪 Бенчмарк производительности PyTorch на M1 Mac")
    print("=" * 50)
    
    # Проверяем доступные устройства
    print("📱 Доступные устройства:")
    print(f"   CPU: ✅")
    print(f"   MPS: {'✅' if torch.backends.mps.is_available() else '❌'}")
    print(f"   CUDA: {'✅' if torch.cuda.is_available() else '❌'}")
    
    # Тестируем CPU
    cpu_time = benchmark_device('cpu')
    
    # Тестируем MPS если доступен
    if torch.backends.mps.is_available():
        mps_time = benchmark_device('mps')
        
        print(f"\n🏆 РЕЗУЛЬТАТЫ:")
        print(f"   CPU время: {cpu_time:.3f} сек")
        print(f"   MPS время: {mps_time:.3f} сек")
        print(f"   🚀 MPS быстрее в {cpu_time/mps_time:.1f}x раз!")
    else:
        print("\n❌ MPS недоступен на этой системе")

if __name__ == "__main__":
    main() 