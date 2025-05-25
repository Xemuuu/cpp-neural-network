#include "MainWindow.h"

template <size_t inputs, size_t hidden_neurons, size_t output_neurons>
MainWindow<inputs, hidden_neurons, output_neurons>::MainWindow() {
    InitWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "MNIST Rysowanie");
    SetTargetFPS(60);

}

template <size_t inputs, size_t hidden_neurons, size_t output_neurons>
bool MainWindow<inputs, hidden_neurons, output_neurons>::DrawButton(int x, int y, int width, int height, const char *text) {
    Rectangle button = { (float)x, (float)y, (float)width, (float)height };
    bool clicked = false;

    if (CheckCollisionPointRec(GetMousePosition(), button)) {
        DrawRectangleRec(button, LIGHTGRAY);
        if (IsMouseButtonPressed(MOUSE_LEFT_BUTTON)) {
            clicked = true;
        }
    } else {
        DrawRectangleRec(button, GRAY);
    }

    DrawRectangleLinesEx(button, 2, BLACK);
    DrawText(text, x + 10, y + 10, 20, BLACK);
    return clicked;
}

template <size_t inputs, size_t hidden_neurons, size_t output_neurons>
void MainWindow<inputs, hidden_neurons, output_neurons>::UpdateGridWithBrush(int mouseX, int mouseY) {
    int x = mouseX / CELL_SIZE;
    int y = mouseY / CELL_SIZE;

    if (x >= 0 && x < GRID_SIZE && y >= 0 && y < GRID_SIZE) {
        for (int offsetY = -1; offsetY <= 1; offsetY++) {
            for (int offsetX = -1; offsetX <= 1; offsetX++) {
                int nx = x + offsetX;
                int ny = y + offsetY;
                if (nx >= 0 && nx < GRID_SIZE && ny >= 0 && ny < GRID_SIZE) {
                    float distance = sqrtf(offsetX * offsetX + offsetY * offsetY);
                    float intensity = 1.0f - distance / 1.5f;
                    if (intensity > 0.0f) {
                        grid[ny][nx] += intensity;
                        if (grid[ny][nx] > 1.0f) grid[ny][nx] = 1.0f;
                    }
                }
            }
        }
    }
}

template <size_t inputs, size_t hidden_neurons, size_t output_neurons>
void MainWindow<inputs, hidden_neurons, output_neurons>::Run() {
    while (!WindowShouldClose()) {
        // Obsługa rysowania myszą
        if (IsMouseButtonDown(MOUSE_LEFT_BUTTON)) {
            Vector2 mousePosition = GetMousePosition();
            UpdateGridWithBrush(mousePosition.x, mousePosition.y);
        }

        // Obsługa przycisków
        if (DrawButton(GRID_SIZE * CELL_SIZE + 20, 50, 160, 50, "Clear")) {
            for (int y = 0; y < GRID_SIZE; y++) {
                for (int x = 0; x < GRID_SIZE; x++) {
                    grid[y][x] = 0.0f;
                }
            }
            showResult = false;
        }

        if (DrawButton(GRID_SIZE * CELL_SIZE + 20, 120, 160, 50, "Recognize")) {
            showResult = true;
        }


        // Rysowanie interfejsu
        BeginDrawing();
        ClearBackground(BLACK);

        // Rysowanie siatki
        for (int y = 0; y < GRID_SIZE; y++) {
            for (int x = 0; x < GRID_SIZE; x++) {
                Color cellColor = { (unsigned char)(grid[y][x] * 255), (unsigned char)(grid[y][x] * 255), (unsigned char)(grid[y][x] * 255), 255 };
                DrawRectangle(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE - 1, CELL_SIZE - 1, cellColor);
            }
        }

        // Rysowanie linii siatki
        for (int i = 0; i <= GRID_SIZE; i++) {
            DrawLine(i * CELL_SIZE, 0, i * CELL_SIZE, GRID_SIZE * CELL_SIZE, DARKGRAY);
            DrawLine(0, i * CELL_SIZE, GRID_SIZE * CELL_SIZE, i * CELL_SIZE, DARKGRAY);
        }

        // Wyświetlanie wyniku po prawej stronie
        if (showResult) {
            const float* pixels = (const float*)grid;
            uint8_t detectedLabel = g_neuralNetwork.ForwardPass(pixels);
            char resultText[20];
            snprintf(resultText, sizeof(resultText), "Result: %d", detectedLabel);
            DrawText(resultText, GRID_SIZE * CELL_SIZE + 20, 200, 30, WHITE);
        }

        EndDrawing();
    }
    CloseWindow();
}