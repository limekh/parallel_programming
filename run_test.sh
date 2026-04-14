#!/bin/bash

echo "========================================"
echo "  Matrix Multiplication - MPI Benchmark"
echo "========================================"
echo ""

# Заголовок CSV
echo "size,processes,time_sec,operations,data_kb" > stats.csv

echo "Starting tests..."
echo ""

# Тесты для разных размеров
for SIZE in 200 400 800 1200; do
    echo "[Size: ${SIZE} x ${SIZE}]"
    
    # Генерация матриц (1 процесс)
    mpirun --allow-run-as-root -n 1 ./matrix_mpi generate input/matrix.txt $SIZE > /dev/null 2>&1
    echo "  Generated: input/matrix.txt"
    
    # Тесты с разным количеством процессов
    for PROCS in 1 2 4 6; do
        # Запускаем MPI, сохраняем только stdout (stderr игнорируем)
        mpirun --allow-run-as-root -n $PROCS ./matrix_mpi run input/matrix.txt output/result.txt 2>/dev/null | tail -1 > temp_output.txt
        
        # Читаем время (последняя строка, третье поле)
        TIME=$(cat temp_output.txt | awk '{print $3}')
        
        # Если время пустое или 0, пробуем альтернативный парсинг
        if [ -z "$TIME" ] || [ "$TIME" = "0" ]; then
            # Пробуем найти строку с размером матрицы
            TIME=$(grep -E "^[0-9]+ [0-9]+ [0-9]" temp_output.txt | awk '{print $3}')
        fi
        
        # Если всё ещё пусто, ставим 0
        if [ -z "$TIME" ]; then
            TIME="0"
        fi
        
        # Вычисляем операции и данные
        OPS=$((SIZE * SIZE * (2 * SIZE - 1)))
        DATA_KB=$((3 * SIZE * SIZE * 8 / 1024))
        
        # Запись в CSV
        echo "${SIZE},${PROCS},${TIME},${OPS},${DATA_KB}" >> stats.csv
        
        echo "  Processes: ${PROCS} | Time: ${TIME} s"
        
        # Верификация (только для 1 процесса)
        if [ $PROCS -eq 1 ]; then
            python3 verify.py input/matrix.txt output/result.txt 2>/dev/null
        fi
    done
    
    # Очистка временного файла
    rm -f temp_output.txt
    
    echo "----------------------------------------"
done

echo ""
echo "Results saved to: stats.csv"
echo ""
