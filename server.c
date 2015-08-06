#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <signal.h>


#define THIS_IS_MASTER 0x007
#define THIS_IS_WORKER 0x004



int main(int argc,char **argv)
{

    unsigned int workerNum=10;
    int who=THIS_IS_MASTER; 
    int n;
    int pid;


   
    /*create worker process */ 
    for(n=0;n<workerNum;n++){

        switch(pid=fork()){
            case -1:
                exit(100);
            case 0: /*child*/
                who =THIS_IS_WORKER;
                break; 
            default:
                who=THIS_IS_MASTER;
                break;
        }
    } 

   
    /*worker process do something*/ 
    if(who == THIS_IS_WORKER){
         int loop =1;

         do{/*woker doing loop*/
            fprintf(stderr,"This is child(%d)\n",getpid());
            sleep(1);
         }while(loop);
    }


   
    /*master process do something*/ 
    if(who == THIS_IS_MASTER){
        printf("this is master process\n");
        sleep(2);         
    }


    exit(0);
}
