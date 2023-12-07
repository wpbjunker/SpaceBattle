#include <pthread.h>
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
#include <time.h>
#include <cuda.h>

#include "driver.h"
#include "board.h"

#define LOST_CONNECTION -2 //clients that the server lost its connection with
#define SERVER_PORT 6689
// directions, used for user input
#define NONE 0
#define UP 1
#define DOWN 2
#define RIGHT 3
#define LEFT 4
#define CANNONBALL_LIMIT 100
/********************************* STRUCTS *********************************/
typedef struct talk_to_client_args {
  int clientID;
  int port;
  int socket;
  spaceship_t * ship;
  bool boosted; //TODO: initialize this somewhere
  int ship_direction; //TODO: initialize this somewhere
  int shoot_direction; //TODO: initialize this somewhere
  bool cannonball_shot; //TODO: initialize this somewhere
} talk_to_client_args_t;

/****************************** GLOBALS ************************************/

// GLOBAL CLIENT LIST
client_list_t * clients;
int client_count;

server_rsp_t * send_to_clients;
pthread_mutex_t send_to_clients_lock;

cannonball_t * cannonballs;
pthread_mutex_t cannonballs_lock;
int num_cannonballs;
pthread_mutex_t continue_flag_lock;
bool continue_flag;

/**************************** FUNCTIONS ************************************/
/*************************** SIGNATURES ************************************/
void stop_game();
void remove_client (int port);
void quit_client (int port);
void end_game ();

/*************************** THREAD FUNCTIONS ******************************/
void client_calculations(talk_to_client_args_t * client_info){
  printf("client_calculations is running\n");
  if(num_cannonballs < CANNONBALL_LIMIT){
    // call functions to handle information
    if(client_info->cannonball_shot){
      if(cannonballs == NULL){
        cannonballs = init_cannonballs(); 
      } // if
      // will the new cannonball be in the bounds of the screen?
      if (is_cannonball_in_bounds(client_info->ship,
                                  client_info->shoot_direction)) {
        pthread_mutex_lock(&cannonballs_lock);
        num_cannonballs += 1; 
        add_cannonball(client_info->ship,
                       client_info->shoot_direction,
                       cannonballs, num_cannonballs);
        pthread_mutex_unlock(&cannonballs_lock);
      }
    } // if a cannonball was shot
  } // if there aren't too many cannonballs shot

  // handle existing cannonballs
  update_cannonballs(cannonballs, num_cannonballs);
  
    // put information together with information about other client
    // step 1: which client are we working with?
  int i = 0;
  while(clients[i].clientID != client_info->clientID);
  printf("clients[i].clientID != client_info->clientID\n");
  // step 2: change the info in send_to_clients for the client you're
  //         working with
  if (i ==0){ // you're working with the first client
    pthread_mutex_lock(&send_to_clients_lock);
    send_to_clients->clientID0 = clients[i].clientID;
    send_to_clients->client_socket0 = clients[i].socket;
    send_to_clients->listen_port0 = clients[i].port_num;
    if(client_info->boosted){
      spaceship_t * ship0 = update_spaceship(client_info->ship,
                                                client_info->ship_direction);
      send_to_clients->ship0_x_position = ship0->x_position;
      send_to_clients->ship0_y_position = ship0->y_position;
      send_to_clients->ship0_x_velocity = ship0->x_velocity;
      send_to_clients->ship0_y_velocity = ship0->y_velocity;

    } // if
    pthread_mutex_unlock(&send_to_clients_lock);
  } else if (i == 1){ // you're working with the second client
    pthread_mutex_lock(&send_to_clients_lock);
    send_to_clients->clientID1 = clients[i].clientID;
    send_to_clients->client_socket1 = clients[i].socket;
    send_to_clients->listen_port1 = clients[i].port_num;
   
    if(client_info->boosted){
      update_spaceship(client_info->ship, client_info->ship_direction);
    } // if
    send_to_clients->ship0_x_position = client_info->ship->x_position;
    send_to_clients->ship0_y_position = client_info->ship->y_position;
    send_to_clients->ship0_x_velocity = client_info->ship->x_velocity;
    send_to_clients->ship0_y_velocity = client_info->ship->y_velocity;

        
    pthread_mutex_unlock(&send_to_clients_lock);
  }
} // client_calculations 

void * talk_to_client(void * args){
  talk_to_client_args_t * client_info = (talk_to_client_args_t *)args;
  //client_info->clientID = client_count;
  
  while(continue_flag){
    // make sure that all the clients are still connected
    for(int i = 0; i < 2; i++){
      if(clients[i].socket == LOST_CONNECTION){
        remove_client(client_info->port);
      } // if
    } // for
    
    // listen for information from client
    msg_to_server * response = (msg_to_server*)
      malloc(sizeof(msg_to_server_t));

    if(send_to_clients->num_changes >= 2){
      // if both clients have given new input
      send_to_clients->num_changes = 0;
      for(int j = 0; j < 2; j++){
        send_to_clients->target_clientID = j;
        write(clients[j].socket, send_to_clients, sizeof(server_rsp_t));
      } // for
    } // if
  } // while
  printf("continue_flag = false\n");
  return NULL;
} // talk_to_client
/************************* END GAME FUNCTIONS ******************************/
void stop_game(){
  server_rsp_t quit_msg;
  printf("Stopping game.\n");
  for(int i = 0; i < 2; i++){
    if(i == 0){
      quit_msg.clientID0 = clients[i].clientID;
      quit_msg.continue_flag = false; // stops client threads
      quit_msg.listen_port0 = clients[i].port_num;
    }
    else{
      quit_msg.clientID1 = clients[i].clientID;
      quit_msg.continue_flag = false; // stops client threads
      quit_msg.listen_port1 = clients[i].port_num;
    }
    write(clients[i].socket, &quit_msg, sizeof(server_rsp_t));
  } // for
  free(clients);
  free(send_to_clients);
  pthread_mutex_destroy(&send_to_clients_lock);
  free_cannonballs(cannonballs); // W! free(cannonballs);
  pthread_mutex_destroy(&cannonballs_lock);

  exit(1);
} // stop_game

// called when a client cannot be communicated with.
void remove_client (int port) {
  printf("Your opponent can't connect to the server.\n");
  stop_game();
} // remove_client

// called when a client quits before the game finishes
void quit_client (int port){
  printf("One player has quit the game.\n");
  stop_game();
} // quit_client

// called when the game ends, which is when at least one player has died
void end_game (){
  stop_game();
} // end_game

/***************************** MAIN ****************************************/

int main() {
  /*============== SET UP: PART 1, SET UP SERVER SOCKET ===================*/
  // Set up a socket
  int s = socket(AF_INET, SOCK_STREAM, 0);
  if(s == -1) {
    perror("socket");
    exit(2);
  }

  // Listen at this address.
  struct sockaddr_in addr;
  memset(&addr, 0, sizeof(addr));
  addr.sin_addr.s_addr = INADDR_ANY;
  addr.sin_family = AF_INET;
  addr.sin_port = htons(SERVER_PORT);
  

  // Bind to the specified address
  if(bind(s, (struct sockaddr*)&addr, sizeof(struct sockaddr_in))) {
    perror("bind");
    exit(2);
  }

  // Become a server socket
  if(listen(s, 2)) {
    perror("listen failed");
    exit(2);
  }
  

  /*=================== SET UP: PART 2, SET UP GLOBALS ====================*/
  client_count = 0;
  // set up the list of connected clients
  clients = (client_list_t*)malloc(sizeof(client_list_t));
  int client_socket;
  send_to_clients = (server_rsp_t *)malloc(sizeof(server_rsp_t));
  
  pthread_mutex_unlock(&continue_flag_lock);
  send_to_clients->continue_flag = true;
  pthread_mutex_lock(&continue_flag_lock);

  send_to_clients->num_changes = 0;
  pthread_mutex_init(&(send_to_clients_lock), NULL);
  pthread_mutex_init(&(cannonballs_lock), NULL);  
  pthread_mutex_init(&(continue_flag_lock), NULL);

  continue_flag = true;
  pthread_mutex_lock(&continue_flag_lock);

  pthread_t new_client_thread;
  /*====================== ACCEPT CLIENT CONNECTIONS ======================*/
  // Accept 2 connections
  while(client_count < 2) {
    printf("Waiting for a client...\n");
    // Accept a client connection
    struct sockaddr_in client_addr;
    socklen_t client_addr_length = sizeof(struct sockaddr_in);
    client_socket = accept(s, (struct sockaddr*)&client_addr,
                           &client_addr_length);
    if(client_socket == -1) {
      perror("accept failed");
      
      exit(2);
    }

    /* STORE SOCKET AND SHIP FOR NEW CLIENT */
    clients[client_count - 1].socket = client_socket;
    clients[client_count - 1].ship = init_spaceship(client_count);
    
    /* LISTEN TO CLIENT */
    msg_to_server_t message;
    if(read(client_socket, &message, sizeof(msg_to_server_t)) == -1){
      // if the server couldn't read from the client, exit the game
      remove_client(message.listen_port);
    }

    /* STORE OTHER INFORMATION ABOUT NEW CLIENT */
    // store the client's ip address 
    char ipstr[INET_ADDRSTRLEN];
    inet_ntop(AF_INET, &client_addr.sin_addr, ipstr, INET_ADDRSTRLEN);

    // Tell the client about itself
    client_list_t* new_client = (client_list_t*)
      malloc(sizeof(client_list_t));
    new_client->clientID = client_count; // clientID
    strncpy(new_client->ip, ipstr, INET_ADDRSTRLEN); // IP
    new_client->port_num = message.listen_port; // port_num
    new_client->socket = clients[client_count-1].socket;
    new_client->ship = init_spaceship(client_count); // ship
    write(new_client->socket, new_client, sizeof(client_list_t));
    
    // Put new client in clients array
    new_client->next = clients; 
    clients = new_client;
    client_count++;
    free(new_client);
    
    /*============ SET UP COMMUNICATION WITH NEW CLIENT ==================*/ 
    // make new thread to communicate with client
    talk_to_client_args_t * args = (talk_to_client_args_t*)
      malloc(sizeof(talk_to_client_args_t));
    args->port = new_client->port_num;
    args->socket = client_socket;
    args->clientID = new_client->clientID;
    args->ship = clients[client_count - 1].ship;
        
    // Thread talks to individual client
    pthread_create(&new_client_thread, NULL, talk_to_client, (void *)(args));

    free(args);
    // end game if necessary
    if (message.continue_flag == false) {
      printf("continue_flag = false\n");
      quit_client(message.listen_port);
    } else {
      // if they aren't trying to quit, connect them
      printf("\nClient %d connected from %s, on port %d\n",
             message.clientID, ipstr, ntohs(message.listen_port));
    } // else
  } // while

  pthread_mutex_unlock(&continue_flag_lock);
  continue_flag = true;
  pthread_mutex_lock(&continue_flag_lock);
  printf("Both clients connected.\n");
  
  pthread_join(new_client_thread, NULL);
}
