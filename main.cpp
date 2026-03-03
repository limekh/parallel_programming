#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>

using namespace std;
using namespace std::chrono;

class MatrixMultiplier {
private:
  vector<vector<double>> matrixA;
  vector<vector<double>> matrixB;
  vector<vector<double>> resultMatrix;
  int size;

public:
  bool readMatrices(const string& filename) {
    ifstream file(filename);
    if (!file.is_open()) {
      cerr << "Error: " << filename << endl;
      return false;
    }
    file >> size;
    matrixA.resize(size, vector<double>(size));
    matrixB.resize(size, vector<double>(size));
    for (int i = 0; i < size; i++)
      for (int j = 0; j < size; j++)
        file >> matrixA[i][j];
    for (int i = 0; i < size; i++)
      for (int j = 0; j < size; j++)
        file >> matrixB[i][j];
    file.close();
    return true;
  }

  void multiplyMatrices() {
    resultMatrix.resize(size, vector<double>(size, 0.0));
    for (int i = 0; i < size; i++)
      for (int j = 0; j < size; j++)
        for (int k = 0; k < size; k++)
          resultMatrix[i][j] += matrixA[i][k] * matrixB[k][j];
  }

  bool writeResult(const string& filename, double execTime) {
    ofstream file(filename);
    if (!file.is_open()) return false;
    
    file << "# Size: " << size << "x" << size << "\n";
    file << "# Time: " << execTime << " s\n";
    file << "# Operations: " << (long long)size * size * (2 * size - 1) << "\n";
    file << "# Data: " << (3 * size * size * sizeof(double)) / 1024.0 << " KB\n\n";
    file << size << "\n";
    for (int i = 0; i < size; i++) {
      for (int j = 0; j < size; j++)
        file << fixed << setprecision(6) << resultMatrix[i][j] << " ";
      file << "\n";
    }
    file.close();
    return true;
  }

  int getSize() const { return size; }
};

void generateMatrices(const string& filename, int size) {
  ofstream file(filename);
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

int main(int argc, char* argv[]) {
  if (argc < 2) {
    cout << "Usage:\n";
    cout << "  " << argv[0] << " generate <file> <size>\n";
    cout << "  " << argv[0] << " run <input> <output>\n";
    return 1;
  }

  string cmd = argv[1];
  
  if (cmd == "generate" && argc >= 4) {
    generateMatrices(argv[2], atoi(argv[3]));
    cout << "Generated: " << argv[2] << "\n";
  }
  else if (cmd == "run" && argc >= 4) {
    MatrixMultiplier m;
    if (!m.readMatrices(argv[2])) return 1;
    
    auto start = high_resolution_clock::now();
    m.multiplyMatrices();
    auto end = high_resolution_clock::now();
    double time = duration<double>(end - start).count();
    
    m.writeResult(argv[3], time);
    cout << m.getSize() << " " << time << "\n";
  }
  else {
    cout << "Unknown command\n";
    return 1;
  }
  
  return 0;
}