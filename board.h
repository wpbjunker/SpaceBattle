
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <SDL.h>

#ifndef __BOARD_H__
#define __BOARD_H__		

/** From gui.h **/
 // Screen size		
#define SCREEN_WIDTH 1024		
#define SCREEN_HEIGHT 768		
		
typedef struct color {		
  uint8_t alpha;		
  uint8_t blue;		
  uint8_t green;		
  uint8_t red;		
} color_t;		

/** From board.h  **/
// Initialize the graphical interface
void gui_init();

// Update the graphical interface to show the latest image data
void gui_update_display();

// Set a single pixel in the image data
void gui_set_pixel(int x, int y, color_t color);

// Add a circle to the image data
void gui_draw_star(int center_x, int center_y, float radius, color_t color);

//Drawing ship (it's a square)
void gui_draw_ship(int center_x, int center_y);

//Drawing cannonballs
void gui_draw_cannonballs(int center_x, int center_y);

// Fade out every pixel by some scale
void gui_fade(float scale);

// Clean up
void gui_shutdown();

#endif
