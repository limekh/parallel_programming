#!/usr/bin/env python3
"""
Построение графиков для CUDA лабораторной работы #4
Адаптировано для формата stats.csv с колонками:
size,block_x,block_y,use_shared,time_ms,operations,data_kb
"""

import csv
import matplotlib.pyplot as plt
import numpy as np

def read_stats(filename='stats.csv'):
    """Читает данные из CSV файла"""
    data = {
        'size': [],
        'block_x': [],
        'block_y': [],
        'use_shared': [],
        'time_ms': [],
        'operations': [],
        'data_kb': []
    }
    
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    data['size'].append(int(row['size']))
                    data['block_x'].append(int(row['block_x']))
                    data['block_y'].append(int(row['block_y']))
                    data['use_shared'].append(int(row['use_shared']))
                    data['time_ms'].append(float(row['time_ms']))
                    data['operations'].append(int(row['operations']))
                    data['data_kb'].append(int(row['data_kb']))
                except (ValueError, KeyError) as e:
                    print(f"Warning: Skipping row - {e}")
                    continue
    except FileNotFoundError:
        print(f"Error: {filename} not found!")
        return None
    
    return data

def plot_time_vs_size(data, output_file='cuda_time_vs_size.png'):
    """График: Время выполнения vs Размер матрицы"""
    plt.figure(figsize=(12, 7))
    
    sizes = sorted(set(data['size']))
    block_configs = sorted(set(zip(data['block_x'], data['block_y'])))
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6', '#1abc9c']
    
    for i, (bx, by) in enumerate(block_configs):
        indices = [j for j in range(len(data['size'])) 
                  if data['block_x'][j] == bx and data['block_y'][j] == by and data['use_shared'][j] == 0]
        
        sizes_subset = [data['size'][j] for j in indices]
        times_subset = [data['time_ms'][j] for j in indices]
        
        if sizes_subset:
            plt.plot(sizes_subset, times_subset, 's-', linewidth=2.5, 
                     color=colors[i % len(colors)], 
                     label=f'{bx}x{by} (global)', markersize=8)
    
    plt.xlabel('Размер матрицы (N x N)', fontsize=12, fontweight='bold')
    plt.ylabel('Время выполнения (мс)', fontsize=12, fontweight='bold')
    plt.title('Зависимость времени выполнения от размера матрицы (CUDA)', 
              fontsize=14, fontweight='bold', pad=15)
    plt.legend(title='Конфигурация блока', loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xticks(sizes)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_file}")

def plot_speedup(data, output_file='cuda_speedup.png'):
    """График: Ускорение относительно базовой конфигурации 8x8 global"""
    plt.figure(figsize=(12, 7))
    
    sizes = sorted(set(data['size']))
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']
    
    for i, size in enumerate(sizes):
        indices = [j for j, s in enumerate(data['size']) if s == size and data['use_shared'][j] == 0]
        
        if not indices:
            continue
            
        base_idx = None
        for j in indices:
            if data['block_x'][j] == 8 and data['block_y'][j] == 8:
                base_idx = j
                break
        
        if base_idx is None:
            continue
            
        base_time = data['time_ms'][base_idx]
        
        configs = [f"{data['block_x'][j]}x{data['block_y'][j]}" for j in indices]
        times = [data['time_ms'][j] for j in indices]
        
        speedup = [base_time / t if t > 0 else 0 for t in times]
        
        x_pos = range(len(configs))
        plt.plot(x_pos, speedup, 'o-', linewidth=2.5, 
                 color=colors[i % len(colors)], 
                 label=f'{size}x{size}', markersize=10)
    
    plt.xlabel('Конфигурация блока', fontsize=12, fontweight='bold')
    plt.ylabel('Ускорение (Speedup)', fontsize=12, fontweight='bold')
    plt.title('Ускорение относительно 8x8 global (CUDA)', 
              fontsize=14, fontweight='bold', pad=15)
    plt.legend(title='Размер матрицы', loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xticks(range(len(configs)), configs, rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_file}")

def plot_shared_vs_global(data, output_file='cuda_shared_vs_global.png'):
    """График: Сравнение shared и global памяти"""
    plt.figure(figsize=(12, 7))
    
    sizes = sorted(set(data['size']))
    colors = ['#3498db', '#2ecc71']
    
    x_positions = range(len(sizes))
    width = 0.35
    
    global_times = []
    shared_times = []
    
    for size in sizes:
        idx_global = [j for j, s in enumerate(data['size']) 
                     if s == size and data['use_shared'][j] == 0 and data['block_x'][j] == 16 and data['block_y'][j] == 16]
        idx_shared = [j for j, s in enumerate(data['size']) 
                     if s == size and data['use_shared'][j] == 1 and data['block_x'][j] == 16 and data['block_y'][j] == 16]
        
        global_times.append(data['time_ms'][idx_global[0]] if idx_global else 0)
        shared_times.append(data['time_ms'][idx_shared[0]] if idx_shared else 0)
    
    plt.bar([x - width/2 for x in x_positions], global_times, width, label='Global', alpha=0.8, color=colors[0])
    plt.bar([x + width/2 for x in x_positions], shared_times, width, label='Shared', alpha=0.8, color=colors[1])
    
    plt.xlabel('Размер матрицы', fontsize=12, fontweight='bold')
    plt.ylabel('Время выполнения (мс)', fontsize=12, fontweight='bold')
    plt.title('Сравнение global и shared памяти (16x16 блок)', 
              fontsize=14, fontweight='bold', pad=15)
    plt.legend(title='Тип памяти', loc='upper right', fontsize=10)
    plt.grid(True, alpha=0.3, axis='y')
    plt.xticks(x_positions, sizes, rotation=0)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_file}")

def plot_gflops(data, output_file='cuda_gflops.png'):
    """График: Производительность в GFLOPS"""
    plt.figure(figsize=(12, 7))
    
    sizes = sorted(set(data['size']))
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']
    
    for i, size in enumerate(sizes):
        indices = [j for j, s in enumerate(data['size']) if s == size and data['use_shared'][j] == 0]
        
        if not indices:
            continue
            
        block_configs = [(data['block_x'][j], data['block_y'][j]) for j in indices]
        times = [data['time_ms'][j] for j in indices]
        ops = [data['operations'][j] for j in indices]
        
        gflops = [(op / (t / 1000 * 1e9)) if t > 0 else 0 for op, t in zip(ops, times)]
        
        configs = [f"{bx}x{by}" for bx, by in block_configs]
        x_pos = range(len(configs))
        
        plt.plot(x_pos, gflops, 's-', linewidth=2.5, 
                 color=colors[i % len(colors)], 
                 label=f'{size}x{size}', markersize=10)
    
    plt.xlabel('Конфигурация блока', fontsize=12, fontweight='bold')
    plt.ylabel('Производительность (GFLOPS)', fontsize=12, fontweight='bold')
    plt.title('Производительность умножения матриц (CUDA)', 
              fontsize=14, fontweight='bold', pad=15)
    plt.legend(title='Размер матрицы', loc='upper right', fontsize=10)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_file}")

def print_analysis(data):
    """Выводит текстовый анализ результатов"""
    print("\n" + "="*80)
    print("АНАЛИЗ РЕЗУЛЬТАТОВ CUDA (GTX 1060 3GB)")
    print("="*80)
    
    sizes = sorted(set(data['size']))
    
    for size in sizes:
        indices = [j for j, s in enumerate(data['size']) if s == size and data['use_shared'][j] == 0]
        
        if not indices:
            continue
            
        print(f"\nМатрица {size}x{size} (global memory):")
        print("-" * 80)
        print(f"{'Блок':<12} {'Потоков':<10} {'Время (мс)':<15} {'GFLOPS':<12}")
        print("-" * 80)
        
        for j in indices:
            bx, by = data['block_x'][j], data['block_y'][j]
            threads = bx * by
            time_ms = data['time_ms'][j]
            ops = data['operations'][j]
            gflops = (ops / (time_ms / 1000 * 1e9)) if time_ms > 0 else 0
            print(f"{bx}x{by:<11} {threads:<10} {time_ms:<15.3f} {gflops:<12.2f}")

def main():
    print("Reading stats.csv...")
    data = read_stats()
    
    if data is None or len(data['size']) == 0:
        print("No data found in stats.csv")
        return 1
    
    print(f"Loaded {len(data['size'])} data points")
    print(f"   Sizes: {sorted(set(data['size']))}")
    print(f"   Block configs: {sorted(set(zip(data['block_x'], data['block_y'])))}")
    print()
    
    print("Generating graphs...")
    print("-" * 80)
    
    plot_time_vs_size(data)
    plot_speedup(data)
    plot_shared_vs_global(data)
    plot_gflops(data)
    
    print()
    print_analysis(data)
    
    print("\n" + "="*80)
    print("ВСЕ ГРАФИКИ СОЗДАНЫ УСПЕШНО!")
    print("="*80)
    print("\nСозданные файлы:")
    print("   1. cuda_time_vs_size.png        - Время vs Размер матрицы")
    print("   2. cuda_speedup.png             - Ускорение (Speedup)")
    print("   3. cuda_shared_vs_global.png    - Shared vs Global память")
    print("   4. cuda_gflops.png              - Производительность (GFLOPS)")
    print("\nОткройте PNG файлы для просмотра графиков")
    print("="*80)
    
    return 0

if __name__ == "__main__":
    exit(main())