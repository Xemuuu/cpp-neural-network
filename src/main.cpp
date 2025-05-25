#include "LoadData.h"
#include "NeuralNetwork.h"
#include "Timer.h"
#include "TrainingNeuralNetwork.h"
#include "MainWindow.h"


// jesli ustawione na 1 zapisuje blad do pliku co moze spowolnic dzialanie programu
#define SAVE_ERROR() 1
 
const size_t c_inputNeurons = 785;
const size_t c_hiddenNeurons = 30; 
const size_t c_outputNeurons = 10;

const size_t c_trainingEpochs = 1;
const size_t c_miniBatchSize = 10;
const float c_learningRate = 3.0f;

// zbiory danych treningowych i testowych
MNISTData trainingData;
MNISTData testData;
 
TrainingNeuralNetwork <c_inputNeurons, c_hiddenNeurons, c_outputNeurons> neuralNetwork;

float calculateAccuracy (const MNISTData& data)
{
    size_t numberOfCorrect = 0;
    for (size_t i = 0, c = data.NumImages(); i < c; ++i)
    {
        uint8_t label;
        const float* pixels = data.GetImage(i, label);
        uint8_t detectedLabel = neuralNetwork.ForwardPass(pixels);

        if (detectedLabel == label)
            ++numberOfCorrect;
    }
    return float(numberOfCorrect) / float(data.NumImages());
}

int runTraining(){

    // załadowujemy dane treningowe i testowe zmienna true/false decyduje o tym czy dane są do treningu czy testowania
    if (!trainingData.Load(true) || !testData.Load(false))
    {
        printf("Nie udalo sie zaladowac danych!\n");
        return 1;
    }
 
    #if SAVE_ERROR()
    FILE *file = fopen("Error.csv","w+t");
    if (!file)
    {
        printf("Nie udalo sie otworzyc pliku Error.csv!\n");
        return 2;
    }
    fprintf(file, "\"Dokladnosc danych treningowych\",\"Dokladnosc danych testowych\"\n");
    #endif

    {
        Timer timer("Czas treningu:  ");
 
        // sprawdznamy dokladnosc przed rozpoczeciem treningu
        for (size_t epoch = 0; epoch < c_trainingEpochs; ++epoch)
        {
            #if SAVE_ERROR()
                float accuracyTraining = calculateAccuracy(trainingData);
                float accuracyTest = calculateAccuracy(testData);
                printf("Dokladnosc danych treningowych: %0.2f%%\n", 100.0f*accuracyTraining);
                printf("Dokladnosc danych testowych: %0.2f%%\n\n", 100.0f*accuracyTest);
                fprintf(file, "\"%f\",\"%f\"\n", accuracyTraining, accuracyTest);
            #endif
 
            printf("Trenowanie %zu / %zu...\n", epoch+1, c_trainingEpochs);
            neuralNetwork.Train(trainingData, c_miniBatchSize, c_learningRate);
            printf("\n");
        }
    }

     
    // na koniec sprawdzamy ostateczna dokladnosc
    float accuracyTraining = calculateAccuracy(trainingData);
    float accuracyTest = calculateAccuracy(testData);
    printf("\nKoncowa dokladnosc danych treningowych: %0.2f%%\n", 100.0f*accuracyTraining);
    printf("Koncowa dokladnosc danych testowych: %0.2f%%\n\n", 100.0f*accuracyTest);
 
    #if SAVE_ERROR()
        fprintf(file, "\"%f\",\"%f\"\n", accuracyTraining, accuracyTest);
        fclose(file);
    #endif
 

    // zapisujemy wagi do pliku    
    {
    FILE* file = fopen("WeightsBiases.txt", "w+t");

    auto hiddenBiases = neuralNetwork.GetHiddenLayerBiases();
    for (size_t i = 0; i < hiddenBiases.size(); ++i)
    {
        fprintf(file, "    %f", hiddenBiases[i]);
        fprintf(file, "\n");
    }
    
    auto hiddenWeights = neuralNetwork.GetHiddenLayerWeights();
    for (size_t i = 0; i < hiddenWeights.size(); ++i)
    {
        fprintf(file, "    %f", hiddenWeights[i]);
        fprintf(file, "\n");
    }

    auto outputBiases = neuralNetwork.GetOutputLayerBiases();
    for (size_t i = 0; i < outputBiases.size(); ++i)
    {
        fprintf(file, "    %f", outputBiases[i]);
        fprintf(file, "\n");
    }

    auto outputWeights = neuralNetwork.GetOutputLayerWeights();
    for (size_t i = 0; i < outputWeights.size(); ++i)
    {
        fprintf(file, "    %f", outputWeights[i]);
        fprintf(file, "\n");
    }

    fclose(file);
    }

    return 0;
}


int main() {
    //runTraining();
    MainWindow<c_inputNeurons, c_hiddenNeurons, c_outputNeurons> mainWindow;
    mainWindow.Run();



    return 0;
}