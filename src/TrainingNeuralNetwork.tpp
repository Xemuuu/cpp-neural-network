#include "TrainingNeuralNetwork.h"

template <size_t inputs, size_t hidden_neurons, size_t output_neurons>
void TrainingNeuralNetwork<inputs, hidden_neurons, output_neurons>::Train(const MNISTData& trainingData, size_t miniBatchSize, float learningRate) {
    // Randomize the order of the training data to create mini-batches
    if (this->m_trainingOrder.size() != trainingData.NumImages())
    {
        this->m_trainingOrder.resize(trainingData.NumImages());
        size_t index = 0;
        for (size_t& v : this->m_trainingOrder)
        {
            v = index;
            ++index;
        }
    }
    static std::random_device rd;
    static std::mt19937 generator(rd());
    std::shuffle(this->m_trainingOrder.begin(), this->m_trainingOrder.end(), generator);

    // Przetwarzamy dane treningowe do momentu, aż wszystkie dane zostaną przetworzone
    size_t trainingIndex = 0;
    while (trainingIndex < trainingData.NumImages())
    {
        // Czyścimy tablice przechowujące sumę kosztów dla wag i biasów przed kolejnym jej wypełnieniem
        std::fill(this->m_miniBatch_ItH_biasesCost.begin(), this->m_miniBatch_ItH_biasesCost.end(), 0.0f);
        std::fill(this->m_miniBatch_HtO_biasesCost.begin(), this->m_miniBatch_HtO_biasesCost.end(), 0.0f);
        std::fill(this->m_miniBatch_ItH_weightsCost.begin(), this->m_miniBatch_ItH_weightsCost.end(), 0.0f);
        std::fill(this->m_miniBatch_HtO_weightsCost.begin(), this->m_miniBatch_HtO_weightsCost.end(), 0.0f);

        // Przetwarzamy dane treningowe w ramach miniBatcha
        size_t miniBatchIndex = 0;
        while (miniBatchIndex < miniBatchSize && trainingIndex < trainingData.NumImages())
        {
            // Pobieramy pojedynczy obraz 
            uint8_t imageLabel = 0;
            const float* pixels = trainingData.GetImage(this->m_trainingOrder[trainingIndex], imageLabel);

            // Znajdujemy poprawną etykietę dla obrazu
            uint8_t labelDetected = this->ForwardPass(pixels);

            // Obliczamy pochodne dla wag i biasów
            BackwardPass(pixels, imageLabel);

            // Dodajemy pochodne do sumy kosztów dla wag i biasów
            for (size_t i = 0; i < this->m_ItH_biasesCost.size(); ++i)
                this->m_miniBatch_ItH_biasesCost[i] += this->m_ItH_biasesCost[i];
            for (size_t i = 0; i < this->m_HtO_biasesCost.size(); ++i)
                this->m_miniBatch_HtO_biasesCost[i] += this->m_HtO_biasesCost[i];
            for (size_t i = 0; i < this->m_ItH_weightsCost.size(); ++i)
                this->m_miniBatch_ItH_weightsCost[i] += this->m_ItH_weightsCost[i];
            for (size_t i = 0; i < this->m_HtO_weightsCost.size(); ++i)
                this->m_miniBatch_HtO_weightsCost[i] += this->m_HtO_weightsCost[i];


            ++trainingIndex;
            ++miniBatchIndex;
        }

        // Aby obliczyc średnią wartość kosztów dla wag i biasów bedziemy musieli podzielić przez ilość obrazów w miniBatchu
        // Zamiast dzielenia kazdego miniBatcha osobno, możemy podzielic wspolczynnik uczenia przez który potem mnożymy
        float miniBatchLearningRate = learningRate / float(miniBatchIndex);

        // Aktualizujemy wagi i biasy odejmujac srednie wartosci kosztow pomnozone przez wspolczynnik uczenia
        // odejmujemy ponieważ srednie wartosci kosztow wskazują kierunek najszybszego wzrostu funkcji kosztu dlatego chcemy poruszac sie w przeciwnym kierunku
        for (size_t i = 0; i < this->m_ItH_biases.size(); ++i)
            this->m_ItH_biases[i] -= this->m_miniBatch_ItH_biasesCost[i] * miniBatchLearningRate;
        for (size_t i = 0; i < this->m_HtO_biases.size(); ++i)
            this->m_HtO_biases[i] -= this->m_miniBatch_HtO_biasesCost[i] * miniBatchLearningRate;
        for (size_t i = 0; i < this->m_ItH_weights.size(); ++i)
            this->m_ItH_weights[i] -= this->m_miniBatch_ItH_weightsCost[i] * miniBatchLearningRate;
        for (size_t i = 0; i < this->m_HtO_weights.size(); ++i)
            this->m_HtO_weights[i] -= this->m_miniBatch_HtO_weightsCost[i] * miniBatchLearningRate;
    }
}


template <size_t inputs, size_t hidden_neurons, size_t output_neurons>
void TrainingNeuralNetwork<inputs, hidden_neurons, output_neurons>::BackwardPass(const float* pixels, uint8_t correctLabel) {
    // zaczynamy od ostatniej warstywy i przechodzimy wstecz
    for (size_t neuronIndex = 0; neuronIndex < output_neurons; ++neuronIndex)
    {
        // Obliczamy wartosci jakie ostatnia warstwa powinna przyjmowac w idealnym przypadku (1 dla poprawnej etykiety, 0 dla pozostalych)
        float wantedOutput = (correctLabel == neuronIndex) ? 1.0f : 0.0f;

        float outputError = this->m_O_outputs[neuronIndex] - wantedOutput;
        float activationDerivative = this->m_O_outputs[neuronIndex] * (1.0f - this->m_O_outputs[neuronIndex]);

        // Obliczamy jak zaktualiować bias dla neuronu
        this->m_HtO_biasesCost[neuronIndex] = outputError * activationDerivative;

        // Obliczamy jak zaktualizować wagi dla neuronu
        for (size_t inputIndex = 0; inputIndex < hidden_neurons; ++inputIndex)
            this->m_HtO_weightsCost[this->HtO_weightIndex(inputIndex, neuronIndex)] = this->m_HtO_biasesCost[neuronIndex] * this->m_H_outputs[inputIndex];
    }

    // dla wag i biasow w warstwie ukrytej dziala to tak samo
    // rozni sie jedynie sposobem obliczania bledu dla neuronu w warstwie ukrytej
    for (size_t neuronIndex = 0; neuronIndex < hidden_neurons; ++neuronIndex)
    {
        // blad dla neuronu w warstwie ukrytej to suma bledow dla kazdego neuronu w warstwie wyjsciowej pomnozonego przez odpowiednia wage
        float outputError = 0.0f;
        for (size_t outputNeuronIndex = 0; outputNeuronIndex < output_neurons; ++outputNeuronIndex)
            outputError += this->m_HtO_biasesCost[outputNeuronIndex] * this->m_HtO_weights[this->HtO_weightIndex(neuronIndex, outputNeuronIndex)];

        float activationDerivative = this->m_H_outputs[neuronIndex] * (1.0f - this->m_H_outputs[neuronIndex]);
        this->m_ItH_biasesCost[neuronIndex] = outputError * activationDerivative;

        for (size_t inputIndex = 0; inputIndex < inputs; ++inputIndex)
            this->m_ItH_weightsCost[this->ItH_weightIndex(inputIndex, neuronIndex)] = this->m_ItH_biasesCost[neuronIndex] * pixels[inputIndex];
    }
}