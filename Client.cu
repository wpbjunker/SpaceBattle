#Include <pthread.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <stdbool.h>
#include <cuda.h>
#include <SDL.h>
#include <errno.h>


#include "driver.h"
#include "board.h"
#define SERVER_PORT 6689

#define NOT_IN_USE -1 // sockets not in use have this value
#define LOST_CONNECTION -2 //clients that the server lost its connection with
// directions, used for user input

#define UP 1
#define DOWN -1
#define RIGHT 2
#define LEFT -2

/******************************** STRUCTS **********************************/
typedef struct user_input_args_t {
  server_rsp_t * response;
  msg_to_server_t * msg_to_server;
  int listen_socket;
} user_input_t;

/****************************** GLOBALS ************************************/

char * server_name;
int connections[2]; // Each index has a socket number
int num_connections;
int global_listen_port;
int global_clientID;
bool global_continue_flag; // True when the client has not quit
pthread_mutex_t connections_lock = PTHREAD_MUTEX_INITIALIZER;
SDL_Renderer* renderer = NULL;

/*********************** FUNCTION SIGNATURES *******************************/

void * listen_relay_func (void * socket_num);
void remove_connection (int index);
int socket_setup (int port, struct sockaddr_in * addr);

/***************************** THREAD FUNCTIONS ****************************/
// thread to listen for and relay messages
void * listen_relay_func (void * socket_num) {
  printf("I'm in listen_relay_func\n");
  int socket = *(int*)(socket_num);
  //free(socket_num);
  while (global_continue_flag) {

    printf("I'm gonna read a new message now.\n");
    server_rsp_t message;
    // try to read a message
    read(socket, &message, sizeof(server_rsp_t));
    if (read(socket, &message, sizeof(server_rsp_t)) < 0) {
        // something went wrong, exit
        printf("Client.cu read failed\n");
        remove_connection(socket);
        break;
        }

      // the information was sent successfully
      
        global_continue_flag = message.continue_flag; 
        gui_draw_ship(message.ship0->x_position,message.ship0->y_position);
        gui_draw_ship(message.ship1->x_position,message.ship1->y_position);
        gui_draw_cannonballs(700, 300);
      
      printf("drawing!\n");
      color_t red = {0,255,0,0};
      gui_draw_star(200,200,50,red);
      // TODO: change other globals' values?
      gui_update_display();      
    }
  printf("global_continue_flag is false\n");
  close(socket);
  return NULL;
} // listen_relay_func


void * user_input_func (void * args){
  user_input_args_t user_input = *(user_input_args_t*)args;
  
  // use arrow keys to move and click to shoot
  SDL_Event events;

  while(global_continue_flag){
    /*========================= HANDLE CLICKS =============================*/
    int mouse_x, mouse_y;
    uint32_t mouse_state = SDL_GetMouseState(&mouse_x, &mouse_y); 
    if(mouse_state & SDL_BUTTON(SDL_BUTTON_LEFT)){
      printf("You clicked!\n");
      user_input.msg_to_server->cannonball_shot = true;
      
      // decide what direction we're shooting in
      if(user_input.response->ship0_x_position > mouse_x){
        // shoot left
        user_input.msg_to_server->shoot_direction = LEFT;
      }
      else if(user_input.response->ship0_x_position < mouse_x){
        // shoot right
        user_input.msg_to_server->shoot_direction = RIGHT;
      }
    } // if the user clicked
      
    /*========================= HANDLE ARROWKEYS ========================*/
    // move ship
    while(SDL_PollEvent(&events) == 1){
      printf("You pressed a key!\n");
      switch(events.type){
      case SDLK_LEFT:
        user_input.msg_to_server->ship_direction = LEFT;
        user_input.msg_to_server->changed = true;
        break;
      case SDLK_RIGHT:
        user_input.msg_to_server->changed = true;
        user_input.msg_to_server->ship_direction = RIGHT;
        break;
      case SDLK_UP:
        user_input.msg_to_server->changed = true;
        user_input.msg_to_server->ship_direction = UP;
        break;
      case SDLK_DOWN:
        user_input.msg_to_server->changed = true;
        user_input.msg_to_server->ship_direction = DOWN;
        break;
      case SDLK_ESCAPE:
        global_continue_flag = false;
        user_input.msg_to_server->continue_flag = false;
        break;
      } // switch
      // SEND MESSAGE TO SERVER
      if(user_input.msg_to_server->changed == true){
        write(user_input.listen_socket, user_input.msg_to_server, sizeof(msg_to_server_t));
      }
    }
  } // while global_continue_flag
  return NULL;
}


void * display_board_func(void* args){
  star_t * stars = init_stars();
  color_t star_color = {0,0,0,255};
  
  gui_draw_star(stars[0].x_position,
                stars[0].y_position,
                stars[0].radius,
                star_color);
  gui_draw_star(stars[1].x_position,
                stars[1].y_position,
                stars[1].radius,
                star_color);
  
  while(global_continue_flag){
    //gui_draw_ship(
    
    //Display the rendered image
    gui_update_display();
 
  } // while global_continue_flag
  return NULL;
}

/*************************** HELPER FUNCTIONS ******************************/
// remove a connection from our list
// MAKE THIS END THE PROGRAM
void remove_connection (int index) {
  printf("remove_connection\n");
  pthread_mutex_lock(&connections_lock);
  connections[index] = LOST_CONNECTION;
  pthread_mutex_unlock(&connections_lock);
  global_continue_flag = false;
  exit(2);
} // remove_connection


// setup a socket to listen for connections (and spin off a listening thread)
int setup_listen() {
  printf("setup_listen\n");
  // set up child socket, which will be constantly listening for incoming
  //  connections
  struct sockaddr_in addr_listen;
  int listen_socket = socket_setup(0, &addr_listen);

  // Bind to the specified address
  if(bind(listen_socket, (struct sockaddr*)&addr_listen,
          sizeof(struct sockaddr_in))) {
    perror("bind");
    exit(2);
  }
  // Start listening on this socket
  if(listen(listen_socket, 2)) {
    perror("listen failed");
    exit(2);
  }
  // Get the listening socket info so we can find out which port we're using
  socklen_t addr_size = sizeof(struct sockaddr_in);
  getsockname(listen_socket, (struct sockaddr *) &addr_listen, &addr_size);
  
  // save the port we're listening on
  global_listen_port = ntohs(addr_listen.sin_port);

  int * listen_socket_num = (int*)malloc(sizeof(int));
  *listen_socket_num = listen_socket;

  return listen_socket;
} // setup_listen

// function to initialize server connection and receive a parent
server_rsp_t * server_connect(msg_to_server_t * client_join) {
  printf("Connecting to server...\n");
  // set up socket to connect to server
  struct sockaddr_in addr;
  int s = socket_setup(SERVER_PORT, &addr);
  printf("a\t");
  // set up the server as passed into the command line

  if(server_name == NULL){
    printf("server_name is NULL\n");
    exit(1);
  }
  struct hostent* server = gethostbyname(server_name);
  printf("b\t");
  if (server == NULL) {
    printf("server is null\n");
    fprintf(stderr, "Unable to find host %s\n", server_name);
    exit(1);
  }
  // Specify the server's address
  bcopy((char*)server->h_addr, (char*)&addr.sin_addr.s_addr,
        server->h_length);
  printf("c\t");

  // Connect to the server
  if(connect(s, (struct sockaddr*)&addr, sizeof(struct sockaddr_in))){
    perror("connect failed");
    printf("ERROR: %s\n", strerror(errno));
    exit(2);
  } // if
  printf("d\n");

  // send client join message (send the port that the client is listening on)
  printf("write = %ld\n", write(s, client_join, sizeof(msg_to_server_t)));

  server_rsp_t * response = (server_rsp_t*)malloc(sizeof(server_rsp_t));
  while( read(s, response, sizeof(server_rsp_t)) == -1);
   
  // return server's response
  return response;
} // server_connect



// function to set up a socket listening to a given port
int socket_setup (int port, struct sockaddr_in * addr) {
  int s = socket(AF_INET, SOCK_STREAM, 0);
  if (s == -1) {
    perror("socket failed");
    exit(2);
  }
  // Set up addresses
  addr->sin_addr.s_addr = INADDR_ANY;
  addr->sin_family = AF_INET;
  addr->sin_port = htons(port);
  return s;
} // socket_setup

/********************************** MAIN ***********************************/

int main(int argc, char**argv){
  /******************** SET UP PART ONE: UI AND GLOBALS  *******************/
  server_name = argv[1];
  global_continue_flag = true;
  msg_to_server_t * msg_to_server = (msg_to_server_t*)
    malloc(sizeof(msg_to_server_t));
  //global_listen_port =
  
  // set up connections array
  for(int i = 0; i < 3; i++){
    connections[i] = NOT_IN_USE;
  } // for

  gui_init();
  

  
  /******** SET UP PART TWO: PREPARE TO RECEIVE CLIENT JOIN REQUESTS *******/
  // set up child socket, which will be constantly listening for incoming
  //   connections
  
  int listen_socket = setup_listen();
  /************************* CONNECT TO SERVER *****************************/
  msg_to_server->clientID = global_clientID;
  msg_to_server->listen_port = global_listen_port; //updated in setup_listen
  msg_to_server->continue_flag = global_continue_flag;

  server_rsp_t * response = server_connect(msg_to_server);
 
  pthread_t thread;
  printf("Creating a thread\n");
  pthread_create(&thread, NULL, listen_relay_func, (void*)&listen_socket);
  // edit our globals to take into account information gotten from the server
  if(response->target_clientID == 0){
    global_clientID = response->clientID0;
  } // if
  else{
    global_clientID = response->clientID1;

  } // else
  
  msg_to_server->clientID = global_clientID;

  /************************* HANDLE USER INPUT *****************************/
  user_input_args_t args;
  args.response = response;
  args.msg_to_server = msg_to_server;
  args.listen_socket = listen_socket;
  pthread_t user_input;
  pthread_create(&user_input, NULL, user_input_func, (void*)&args);

 
  pthread_join(thread, NULL);
  
  // Free up space
  //free(msg_to_server);
  free(response);
  //free(stars);
  close(listen_socket);

  //Clean up the graphical interface
  gui_shutdown();
  
} // main
