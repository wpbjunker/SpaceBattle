#include "board.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <SDL.h>

#define CANNONBALL_RADIUS 5
#define SHIP_RADIUS 10

SDL_Window* sdlwindow;
SDL_Renderer* sdlrenderer;
SDL_Texture* sdltexture;
color_t bitmap[SCREEN_WIDTH * SCREEN_HEIGHT];

// Initialize the graphical interface
void gui_init() {
  // Initialize the SDL library
  if(SDL_Init(SDL_INIT_VIDEO) != 0) {
    fprintf(stderr, "Failed to initialize SDL: %s\n", SDL_GetError());
    exit(2);
  }
  
  // Create SDL window
  sdlwindow = SDL_CreateWindow("Space battle!",
                               SDL_WINDOWPOS_UNDEFINED,
                               SDL_WINDOWPOS_UNDEFINED,
                               SCREEN_WIDTH,
                               SCREEN_HEIGHT,
                               0);
  if(sdlwindow == NULL) {
    fprintf(stderr, "Failed to create SDL window: %s\n", SDL_GetError());
    SDL_Quit(); // Shut down SDL
    exit(2);
  }
  
  // Create an SDL renderer
  sdlrenderer = SDL_CreateRenderer(sdlwindow, -1, 0);
  if(sdlrenderer == NULL) {
    fprintf(stderr, "Failed to create SDL renderer: %s\n", SDL_GetError());
    SDL_DestroyWindow(sdlwindow);
    SDL_Quit();
    exit(2);
  }
  
  // Clear the display
  SDL_SetRenderDrawColor(sdlrenderer, 255, 255, 255, 255);
  SDL_RenderClear(sdlrenderer);
  
  // Create a texture that the renderer will display
  sdltexture = SDL_CreateTexture(sdlrenderer,
                                 SDL_PIXELFORMAT_RGBA8888,
                                 SDL_TEXTUREACCESS_STREAMING,
                                 SCREEN_WIDTH,
                                 SCREEN_HEIGHT);
                               
  if(sdltexture == NULL) {
    fprintf(stderr, "Failed to create SDL texture: %s\n", SDL_GetError());
    SDL_DestroyRenderer(sdlrenderer);
    SDL_DestroyWindow(sdlwindow);
    SDL_Quit();
    exit(2);
  }
  
  // Clear the bitmap the GUI displays
  memset(bitmap, 0, sizeof(color_t) * SCREEN_WIDTH * SCREEN_HEIGHT);
  gui_update_display();
  SDL_RenderPresent(sdlrenderer);

}

// Update the graphical interface to show the latest image data
void gui_update_display() {
  // Get the raw texture data from SDL
  uint32_t* texture_data;
  int pitch;
  SDL_LockTexture(sdltexture, NULL, (void**)&texture_data, &pitch);

  // Copy in our bitmap data
  memcpy(texture_data, bitmap, sizeof(color_t) * SCREEN_WIDTH * SCREEN_HEIGHT);

  // Release the raw texture data
  SDL_UnlockTexture(sdltexture);
  gui_draw_ship(100, 100);

  // Copy the texture to the renderer
  SDL_Rect destination = { 0, 0, SCREEN_WIDTH, SCREEN_HEIGHT };
  SDL_RenderCopy(sdlrenderer, sdltexture, NULL, &destination);
  
  // Display the bitmap
  SDL_RenderPresent(sdlrenderer);
}

// Set a single pixel in the image data
void gui_set_pixel(int x, int y, color_t color) {
  if(x >= 0 && x < SCREEN_WIDTH && y >= 0 && y < SCREEN_HEIGHT) {
    bitmap[y * SCREEN_WIDTH + x] = color;
  }
}

// Add a circle to the image data
void gui_draw_star(int center_x, int center_y, float radius, color_t color) {
  // Loop over one quadrant of the circle's points
  // Based on method from http://groups.csail.mit.edu/graphics/classes/6.837/F98/Lecture6/circle.html
  for(int x=0; x<=radius; x++) {
    for(int y=0; y<=radius; y++) {
      // If this point is inside the circle, draw a point in all four quadrants
      if(x * x + y * y < radius * radius) {
        gui_set_pixel(center_x + x, center_y + y, color);
        gui_set_pixel(center_x + x, center_y - y, color);
        gui_set_pixel(center_x - x, center_y + y, color);
        gui_set_pixel(center_x - x, center_y - y, color);
      }
    }
  }
}

//Drawing ship (it's a square)
void gui_draw_ship(int center_x, int center_y){
  // Based on method from http://groups.csail.mit.edu/graphics/classes/6.837/F98/Lecture6/circle.html
  color_t ship_color = {0,0,255,0};
  for(int x=0; x<=SHIP_RADIUS; x++) {
    for(int y=0; y<=SHIP_RADIUS; y++) {
      gui_set_pixel(center_x + x, center_y + y, ship_color);
      gui_set_pixel(center_x + x, center_y - y, ship_color);
      gui_set_pixel(center_x - x, center_y + y, ship_color);
      gui_set_pixel(center_x - x, center_y - y, ship_color);
    }
  }
}


//Drawing cannonballs
void gui_draw_cannonballs(int center_x, int center_y){
  // Loop over one quadrant of the circle's points
  // Based on method from http://groups.csail.mit.edu/graphics/classes/6.837/F98/Lecture6/circle.html
  color_t cannonball_color = {0,0,0,255};
  for(int x=0; x<=CANNONBALL_RADIUS; x++) {
    for(int y=0; y<=CANNONBALL_RADIUS; y++) {
      // If this point is inside the circle, draw a point in all four quadrants
      if(x * x + y * y < CANNONBALL_RADIUS * CANNONBALL_RADIUS) {
        gui_set_pixel(center_x + x, center_y + y, cannonball_color);
        gui_set_pixel(center_x + x, center_y - y, cannonball_color);
        gui_set_pixel(center_x - x, center_y + y, cannonball_color);
        gui_set_pixel(center_x - x, center_y - y, cannonball_color);
      }
    }
  }
}
// Fade out every pixel by some scale
void gui_fade(float scale) {
  for(int i=0; i<SCREEN_WIDTH * SCREEN_HEIGHT; i++) {
    bitmap[i].red *= scale;
    bitmap[i].green *= scale;
    bitmap[i].blue *= scale;
  }
}

// Clean up
void gui_shutdown() {
  // Clean up before exiting
  SDL_DestroyTexture(sdltexture);
  SDL_DestroyRenderer(sdlrenderer);
  SDL_DestroyWindow(sdlwindow);
  SDL_Quit();
}
