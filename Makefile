CXX = mpicxx
CXXFLAGS = -O3 -std=c++11
TARGET = matrix_mpi
SRC = main_mpi.cpp

.PHONY: all clean test verify full-test install-deps

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $@ $<

# Генерация тестовых матриц
generate: $(TARGET)
	mpirun --allow-run-as-root -n 1 ./$(TARGET) generate input/matrix.txt 200

# Одиночный запуск
run: $(TARGET)
	mpirun --allow-run-as-root -n 1 ./$(TARGET) run input/matrix.txt output/result.txt

# Тестирование с разным количеством процессов
test: $(TARGET)
	bash run_test.sh

# Верификация
verify:
	python3 verify.py input/matrix.txt output/result.txt

# Полное тестирование с графиками
full-test: $(TARGET)
	bash run_test.sh
	python3 plot_results.py

# Установка зависимостей
install-deps:
	sudo apt install python3-numpy python3-matplotlib -y

# Очистка
clean:
	rm -f $(TARGET)
	rm -rf input/*.txt output/*.txt
	rm -f stats.csv *.png temp_output.txt

# Создание директорий
init:
	mkdir -p input output

# Быстрый тест
quick-test: $(TARGET)
	mpirun --allow-run-as-root -n 1 ./$(TARGET) generate input/matrix.txt 100
	mpirun --allow-run-as-root -n 2 ./$(TARGET) run input/matrix.txt output/result.txt
	python3 verify.py input/matrix.txt output/result.txt
