#include <NeuralNetwork.h>

template <size_t inputs, size_t hidden_neurons, size_t output_neurons>
NeuralNetwork<inputs, hidden_neurons, output_neurons>::NeuralNetwork() {
    generateWeightsAndBiases();
    loadWeights();
}

template <size_t inputs, size_t hidden_neurons, size_t output_neurons>
void NeuralNetwork<inputs, hidden_neurons, output_neurons>::generateWeightsAndBiases() {
    std::random_device rd;
    std::mt19937 generator(rd());
    std::normal_distribution<float> dist(0, 1);

    for (float& f : m_ItH_biases)
        f = dist(generator);

    for (float& f : m_HtO_biases)
        f = dist(generator);

    for (float& f : m_ItH_weights)
        f = dist(generator);

    for (float& f : m_HtO_weights)
        f = dist(generator);
}

template <size_t inputs, size_t hidden_neurons, size_t output_neurons>
uint8_t NeuralNetwork<inputs, hidden_neurons, output_neurons>::ForwardPass(const float* pixels) {
    for (size_t neuronIndex = 0; neuronIndex < hidden_neurons; ++neuronIndex) {
        // pobieramy wartość biasu dla neuronu
        float Z = m_ItH_biases[neuronIndex];

        // do biasu dodajemy wartości pikseli pomnożone przez wagi
        for (size_t inputIndex = 0; inputIndex < inputs; ++inputIndex)
            Z += pixels[inputIndex] * m_ItH_weights[ItH_weightIndex(inputIndex, neuronIndex)];

        // zapisujemy znormalizowaną wartość otrzymaną z funkcji sigmoidalnej
        m_H_outputs[neuronIndex] = 1.0f / (1.0f + std::exp(-Z));
    }

    // powtarzamy proces dla warstwy wyjściowej
    for (size_t neuronIndex = 0; neuronIndex < output_neurons; ++neuronIndex)
    {
        float Z = m_HtO_biases[neuronIndex];
 
        for (size_t inputIndex = 0; inputIndex < hidden_neurons; ++inputIndex)
            Z += m_H_outputs[inputIndex] * m_HtO_weights[HtO_weightIndex(inputIndex, neuronIndex)];
 
        m_O_outputs[neuronIndex] = 1.0f / (1.0f + std::exp(-Z));
    }

    // zwracamy indeks neuronu z największą wartością wyjściową
    float maxOutput = m_O_outputs[0];
    uint8_t maxLabel = 0;
    for (uint8_t neuronIndex = 1; neuronIndex < output_neurons; ++neuronIndex)
    {
        if (m_O_outputs[neuronIndex] > maxOutput)
        {
            maxOutput = m_O_outputs[neuronIndex];
            maxLabel = neuronIndex;
        }
    }
    
    return maxLabel;
}

template <size_t inputs, size_t hidden_neurons, size_t output_neurons>
int NeuralNetwork<inputs, hidden_neurons, output_neurons>::loadWeights() {
    // funkcja odczytuje wagi z pliku w podobny sposób jak są one zapisywane 
    FILE* file = fopen("WeightsBiases.txt", "r+t");
    if (!file)
        return 1;

    for (size_t i = 0; i < hidden_neurons; ++i)
        fscanf(file, "    %f", &m_ItH_biases[i]);

    for (size_t i = 0; i < inputs * hidden_neurons; ++i)
        fscanf(file, "    %f", &m_ItH_weights[i]);

    for (size_t i = 0; i < output_neurons; ++i)
        fscanf(file, "    %f", &m_HtO_biases[i]);

    for (size_t i = 0; i < hidden_neurons * output_neurons; ++i)
        fscanf(file, "    %f", &m_HtO_weights[i]);

    fclose(file);
    return 0;
}