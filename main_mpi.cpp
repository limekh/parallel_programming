#include <mpi.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <string>
#include <cstdlib>

using namespace std;

int procRank = 0;
int procNum = 1;

// Генерация тестовых матриц (только процесс 0)
void generateMatrices(const string& filename, int size) {
    if (procRank != 0) return;
    
    ofstream file(filename);
    if (!file.is_open()) {
        cerr << "Error creating file: " << filename << endl;
        return;
    }
    
    file << size << "\n";
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++)
            file << (i + j + 1) << " ";
        file << "\n";
    }
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++)
            file << (i * j + 1) << " ";
        file << "\n";
    }
    file.close();
}

// MPI умножение матриц
void matrixMultiplicationMPI(vector<double>& matrixA, vector<double>& matrixB,
                             vector<double>& result, int size) {
    
    // Вычисляем количество строк для каждого процесса
    int baseRows = size / procNum;
    int remainder = size % procNum;
    
    // Количество строк для текущего процесса
    int myRows = baseRows + (procRank < remainder ? 1 : 0);
    
    // Вычисляем смещение для текущего процесса
    int myStartRow = 0;
    for (int i = 0; i < procRank; i++) {
        myStartRow += baseRows + (i < remainder ? 1 : 0);
    }
    
    // Выделяем память для локальных буферов
    vector<double> localA(myRows * size);
    vector<double> localResult(myRows * size, 0.0);
    
    // Копируем свои строки из матрицы A в локальный буфер
    for (int i = 0; i < myRows; i++) {
        for (int j = 0; j < size; j++) {
            localA[i * size + j] = matrixA[(myStartRow + i) * size + j];
        }
    }
    
    // Рассылаем матрицу B всем процессам
    MPI_Bcast(matrixB.data(), size * size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // Каждый процесс вычисляет свою часть результата
    for (int i = 0; i < myRows; i++) {
        for (int j = 0; j < size; j++) {
            double sum = 0.0;
            for (int k = 0; k < size; k++) {
                sum += localA[i * size + k] * matrixB[k * size + j];
            }
            localResult[i * size + j] = sum;
        }
    }
    
    // Подготавливаем counts и displs для Gatherv
    vector<int> recvCounts(procNum);
    vector<int> displs(procNum);
    
    for (int i = 0; i < procNum; i++) {
        int rows = baseRows + (i < remainder ? 1 : 0);
        recvCounts[i] = rows * size;
        displs[i] = 0;
        if (i > 0) {
            displs[i] = displs[i-1] + recvCounts[i-1];
        }
    }
    
    // Собираем результаты на процессе 0
    if (procRank == 0) {
        result.resize(size * size);
    }
    
    MPI_Gatherv(localResult.data(), myRows * size, MPI_DOUBLE,
                result.data(), recvCounts.data(), displs.data(),
                MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

// Запись результата (только процесс 0)
bool writeResult(const string& filename, const vector<double>& result, 
                 int size, double execTime) {
    if (procRank != 0) return true;
    
    ofstream file(filename);
    if (!file.is_open()) return false;
    
    file << "# Size: " << size << "x" << size << "\n";
    file << "# Processes: " << procNum << "\n";
    file << "# Time: " << execTime << " s\n";
    file << "# Operations: " << (long long)size * size * (2 * size - 1) << "\n";
    file << "# Data: " << (3 * size * size * sizeof(double)) / 1024.0 << " KB\n\n";
    file << size << "\n";
    
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++)
            file << fixed << setprecision(6) << result[i * size + j] << " ";
        file << "\n";
    }
    file.close();
    return true;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &procRank);
    MPI_Comm_size(MPI_COMM_WORLD, &procNum);
    
    if (argc < 2) {
        if (procRank == 0) {
            cout << "Usage:\n";
            cout << "  mpirun -n <procs> ./matrix_mpi generate <file> <size>\n";
            cout << "  mpirun -n <procs> ./matrix_mpi run <input> <output>\n";
        }
        MPI_Finalize();
        return 1;
    }
    
    string cmd = argv[1];
    
    if (cmd == "generate" && argc >= 4) {
        generateMatrices(argv[2], atoi(argv[3]));
        if (procRank == 0) {
            cout << "Generated: " << argv[2] << " (" << atoi(argv[3]) << "x" << atoi(argv[3]) << ")" << endl;
        }
    }
    else if (cmd == "run" && argc >= 4) {
        string inputFile = argv[2];
        string outputFile = argv[3];
        
        int size = 0;
        vector<double> matrixA, matrixB, result;
        
        // Процесс 0 читает размер из файла
        if (procRank == 0) {
            ifstream file(inputFile);
            if (!file.is_open()) {
                cerr << "Error opening file: " << inputFile << endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            file >> size;
            file.close();
            
            // Выделяем память и читаем матрицы
            matrixA.resize(size * size);
            matrixB.resize(size * size);
            
            ifstream inFile(inputFile);
            inFile >> size; // пропускаем размер
            for (int i = 0; i < size * size; i++)
                inFile >> matrixA[i];
            for (int i = 0; i < size * size; i++)
                inFile >> matrixB[i];
            inFile.close();
        }
        
        // Рассылаем размер всем процессам
        MPI_Bcast(&size, 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        // Выделяем память на всех процессах ПОСЛЕ получения размера
        if (procRank != 0) {
            matrixA.resize(size * size);
            matrixB.resize(size * size);
        }
        
        // Рассылаем матрицу A всем процессам (нужна для корректного Scatterv)
        MPI_Bcast(matrixA.data(), size * size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        
        // Синхронизация перед замером времени
        MPI_Barrier(MPI_COMM_WORLD);
        
        // Замер времени
        double startTime = MPI_Wtime();
        matrixMultiplicationMPI(matrixA, matrixB, result, size);
        double endTime = MPI_Wtime();
        
        double execTime = endTime - startTime;
        
        MPI_Barrier(MPI_COMM_WORLD);
        
        // Запись результата (только процесс 0)
        if (procRank == 0) {
            writeResult(outputFile, result, size, execTime);
            cout << size << " " << procNum << " " << fixed << execTime << endl;
            cout.flush();
        }
    }
    
    MPI_Finalize();
    return 0;
}
