#pragma once

#include "NeuralNetwork.h"


template <size_t inputs, size_t hidden_neurons, size_t output_neurons>
class TrainingNeuralNetwork : public NeuralNetwork<inputs, hidden_neurons, output_neurons> {


public:
    void Train(const MNISTData& trainingData, size_t miniBatchSize, float learningRate);
    void BackwardPass (const float* pixels, uint8_t correctLabel);
};


#include "TrainingNeuralNetwork.tpp"