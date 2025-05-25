#pragma once
#include <iostream>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <random>
#include <array>
#include <vector>
#include <algorithm>
#include <LoadData.h>




template <size_t inputs, size_t hidden_neurons, size_t output_neurons>
class NeuralNetwork {
protected:
    // Wagi i biasy dla warstwy ukrytej i wyjściowej
    std::array<float, inputs * hidden_neurons>          m_ItH_weights;
    std::array<float, hidden_neurons * output_neurons>  m_HtO_weights;
    std::array<float, hidden_neurons>                   m_ItH_biases;
    std::array<float, output_neurons>                   m_HtO_biases;
    
    // Wyjścia z warstw ukrytej i wyjściowej
    std::array<float, hidden_neurons>                   m_H_outputs;
    std::array<float, output_neurons>                   m_O_outputs;
 
    // Koszty dla wag i biasów
    std::array<float, hidden_neurons>                   m_ItH_biasesCost;
    std::array<float, output_neurons>                   m_HtO_biasesCost;
    std::array<float, inputs * hidden_neurons>          m_ItH_weightsCost;
    std::array<float, hidden_neurons * output_neurons>  m_HtO_weightsCost;
 
    // Suma kosztów dla wag i biasów
    std::array<float, hidden_neurons>                   m_miniBatch_ItH_biasesCost;
    std::array<float, output_neurons>                   m_miniBatch_HtO_biasesCost;
    std::array<float, inputs * hidden_neurons>          m_miniBatch_ItH_weightsCost;
    std::array<float, hidden_neurons * output_neurons>  m_miniBatch_HtO_weightsCost;
 
    std::vector<size_t>                                 m_trainingOrder;

    static size_t ItH_weightIndex (size_t inputIndex, size_t hiddenIndex)
    {                                                                               // funkcja zwraca indeks wagi dla danego neuronu w warstwie ukrytej
        return hiddenIndex * inputs + inputIndex;                                   // i danego neuronu w warstwie wejściowej
    }                                                           

    static size_t HtO_weightIndex (size_t hiddenIndex, size_t outputIndex)
    {                                                                               // funkcja zwraca indeks wagi dla danego neuronu w warstwie wyjściowej
        return outputIndex * hidden_neurons + hiddenIndex;                          // i danego neuronu w warstwie ukrytej
    }

public:
    NeuralNetwork();
    uint8_t ForwardPass(const float* pixels);
    void generateWeightsAndBiases();
    int loadWeights();


    const std::array<float, hidden_neurons>& GetHiddenLayerBiases () const { return m_ItH_biases; }
    const std::array<float, output_neurons>& GetOutputLayerBiases () const { return m_HtO_biases; }
    const std::array<float, inputs * hidden_neurons>& GetHiddenLayerWeights () const { return m_ItH_weights; }
    const std::array<float, hidden_neurons * output_neurons>& GetOutputLayerWeights () const { return m_HtO_weights; }
};


#include "NeuralNetwork.tpp"