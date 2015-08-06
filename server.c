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
    sigset_t           set;
    int pid;

    /**
    *设置信号处理:包括需要处理那些信号及处理方法
    */
    //_signal_init(signals);

    /**
    *然后对这些设置的信号进行堵塞，这样做的原因有：
    *1 ，这样下面创建的子进程就不会接收外部发送的信号
    *2 ，在处理以下临界代码的时候不会被信号中断
    */

    sigemptyset(&set);
    sigaddset(&set, SIGCHLD);
    sigaddset(&set, SIGHUP);
    sigprocmask(SIG_BLOCK, &set, NULL);

    sigemptyset(&set);


    /*create worker process */ 
    while(workerNum>0 && who == THIS_IS_MASTER){

        switch(pid=fork()){
            case -1:
                exit(100);
            case 0: /*child*/
                who =THIS_IS_WORKER;
                break; 
            default:
                workerNum--;
                break;
        }
    } 

   
    /*worker process do something*/ 
    if(who == THIS_IS_WORKER){
         int worker_loop =1;

        /**不堵塞任何信号...之前被父进程设置为堵塞,所以这里放开*/
        /**
        sigemptyset(&set);
        if (sigprocmask(SIG_SETMASK, &set, NULL) == -1) {
                ...
        }
        */

         do{/*woker doing loop*/
    
            fprintf(stderr,"This is child(%d)\n",getpid());
            sleep(1);

         }while(worker_loop);
    }


   
    /*master process do something*/ 
    if(who == THIS_IS_MASTER){
        int master_loop =1;

        while(master_loop){

            printf("this is master-----\n");
    
            /**
            *这里可以设置超时代码，用来定时检查相关任务
            */

        

            /**
            *父进程完成上面的任务后，恢复信号处理。
            *堵塞在下面的sigsuspend函数，除非接收到set设置的信号
            */                                   
            sigsuspend(&set);


            /*接收到外部信号后，负责通知子进程*/
           // mastr_child_channel();

        }
    }



    exit(0);
}
