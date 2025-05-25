#pragma once
#include "raylib.h"
#include "NeuralNetwork.h"



const int GRID_SIZE = 28;   // Wymiary siatki do ryswania
const int CELL_SIZE = 20;   // Rozmiar pojedynczej komórki 
const int SCREEN_WIDTH = GRID_SIZE * CELL_SIZE + 200; 
const int SCREEN_HEIGHT = GRID_SIZE * CELL_SIZE;


template <size_t inputs, size_t hidden_neurons, size_t output_neurons>
class MainWindow {
    float grid[GRID_SIZE][GRID_SIZE] = {0.0f};
    bool showResult = false;

    bool DrawButton(int x, int y, int width, int height, const char *text);
    void UpdateGridWithBrush(int mouseX, int mouseY);
    NeuralNetwork <inputs, hidden_neurons, output_neurons> g_neuralNetwork;


public:
	MainWindow();
    void Run();

};



#include "MainWindow.tpp"