#include <iostream>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <stdlib.h> 
#include <mpi.h>
#include <time.h>

using namespace std;

int main(int argc, char* argv[]){

  int rank, size, ierr;
  MPI_Comm comm;
  comm  = MPI_COMM_WORLD;
  MPI_Init(NULL,NULL);
  MPI_Comm_rank(comm, &rank);    
  MPI_Comm_size(comm, &size);
  MPI_Request request;   // for non-blicking send-receive later on
  MPI_Status status;

  int M = 2305;  // M length intervals.
  int Jn = ((M+1)-2)/size + 2;
  double T = 1;  // final time.
  double dt = 0.2/M;
  int N = T/dt;
  int t1=0.333/dt, t2=0.666/dt, t3=N; // points at which we want to print our results.
  double dy = 2./M;
  double dy2 = dy*dy;
  double dtdy2 = dt*dt/dy2;



// Dynamically allocating local Ul array
  double*** Ul = new double** [3]; // this creates an array of size 3 which is ready do hold 2D arrays in each entry.
  for (int i = 0; i < 3; ++i) {   // this loop puts an array of size M+1 on each of the three entries.
    Ul[i] = new double*[Jn];		  // we get a 2D array.
    for (int j = 0; j < Jn; ++j){ // put another array on each entry of the 2D array to get a 3D array.
      Ul[i][j] = new double[M+1];
    }
  }

// Dynamilally allocating local Ul array
  double** U = new double* [M+1]; // this creates an array of size 3 which is ready do hold 2D arrays in each entry.
  for (int i = 0; i < M+1; ++i) {   // this loop puts an array of size M+1 on each of the three entries.
    U[i] = new double[M+1];		  // we get a 2D array.
  }

// initialize numerical array with given conditions.
  for (int i=0; i<Jn; ++i){       // set exponential initialization for t=0,1 all over the internal domain 
  	for (int j=1; j<M; ++j){
  		Ul[0][i][j] = Ul[1][i][j] = exp( -40 * ( ((rank*(Jn-2)+i)*dy-1-0.4)*((rank*(Jn-2)+i)*dy-1-0.4) + (j*dy-1)*(j*dy-1) ) );
  	}
  }
  for (int i=0; i<Jn; ++i){    // set dirichlet zero boundary conditions for t=0,1,2 
  	Ul[0][i][0] = Ul[0][i][M] = Ul[1][i][0] = Ul[1][i][M] = Ul[2][i][0] = Ul[2][i][M] = 0; // 1st & last column Dirichlet b.c.	
  }

  if (rank == 0){ 
     for (int i=0; i<=M; ++i){    // set dirichlet zero boundary conditions for t=0,1,2 
       	  Ul[0][0][i] =  Ul[1][0][i] = Ul[2][0][i] = 0;  // 1st row Dirichlet B.C.
     }
  }

  if (rank == size-1){
     for (int i=0; i<=M; ++i){    // set dirichlet zero boundary conditions for t=0,1,2 
          Ul[0][Jn-1][i] = Ul[1][Jn-1][i] = Ul[2][Jn-1][i]=0; // set last row to Dirichlet B.Cs 
     }
  }
 

// ---------------- printing time zero U values into file ----------------------

// Gathering everything back into array U
  if (rank==0){
     for (int m=0; m<Jn; m++){
         for (int n=0; n<=M; n++){
              U[m][n] = Ul[0][m][n];
         } 
     }
  }


  if (rank!=0){
     MPI_Send(&Ul[0][0][0],Jn*(M+1), MPI_DOUBLE, 0, 2, comm);
  }


  if (rank==0){
     for (int r=1; r<=size-1; r++){
         MPI_Recv(&U[r*(Jn-2)][0],Jn*(M+1), MPI_DOUBLE, r, 2, comm, MPI_STATUS_IGNORE);
     }
  }

  if (rank == 0){
     // print initial U values to file, row by row.
     ofstream out {"U_t0.csv"};
     out<<fixed<<setprecision(4);
     for(int i=0; i<=M; ++i){
         for(int j=0; j<=M; ++j){    
             out<<U[i][j]<<" ";
         }   
     out<<endl;
     }    
  out.close();
  }		


// ---------------------------------------------------------------

  double t_one =  MPI_Wtime();
// use numerical scheme to obtain the future values of U.
  for (int t=1; t<=N; ++t){
  	for (int i=1; i<Jn-1; ++i){
  		for (int j=1; j<M; ++j){		
  			Ul[2][i][j] = 2*Ul[1][i][j] - Ul[0][i][j]	 
                        + dtdy2*( Ul[1][i+1][j] + Ul[1][i-1][j] + Ul[1][i][j+1] + Ul[1][i][j-1] - 4*Ul[1][i][j] ); 	
  		}		
  	}

  if (rank !=0){    // each process apart from the 1st one sends its 2nd row to the previous one
      MPI_Send(&Ul[2][1][0],M+1, MPI_DOUBLE, rank-1, 2, comm);  // Ul[time][row][column] start at row 1(2nd) and send all cols starting from 0
  }

  if (rank != size-1){  // each process apart from the last one receives 
                        // the second row sent by the next process
      MPI_Recv(&Ul[2][Jn-1][0],M+1, MPI_DOUBLE, rank+1, 2, comm, MPI_STATUS_IGNORE);
  }   

  if (rank !=size-1){    // each process apart from the last one sends
                         // its previous to last row to the next one
      MPI_Send(&Ul[2][Jn-2][0],M+1, MPI_DOUBLE, rank+1, 2, comm);
  }   

  if (rank != 0){  // each process apart from the first one receives 
                   // the previous to last row sent by the previous process
      MPI_Recv(&Ul[2][0][0],M+1, MPI_DOUBLE, rank-1, 2, comm, MPI_STATUS_IGNORE);
  } 


// update the previous times.
  for (int i=0; i<=Jn-1; ++i){
      for (int j=0; j<=M; ++j){		
           Ul[0][i][j] = Ul[1][i][j];
           Ul[1][i][j] = Ul[2][i][j];
      }		
  }
  
 // print out files for fixed times	
  	if(t==t1 || t==t2 || t==t3){	
        	
        // Gathering everything back into array U
            if (rank==0){
               for (int m=0; m<Jn; m++){
                   for (int n=0; n<=M; n++){
                        U[m][n] = Ul[1][m][n];
                   } 
               }
            }

            if (rank!=0){
               MPI_Send(&Ul[2][0][0],Jn*(M+1), MPI_DOUBLE, 0, 2, comm);
            }
  
            if (rank==0){
               for (int r=1; r<=size-1; r++){
                   MPI_Recv(&U[r*(Jn-2)][0],Jn*(M+1), MPI_DOUBLE, r, 2, comm, MPI_STATUS_IGNORE);
               }
            }
            if (rank == 0){
        	stringstream ss;
  		ss << fixed << setprecision(2) << t*dt; // this ensures that the double value gets converted
  		string time = ss.str();						 // to string with only 2 trailing digits.
  		
  		ofstream out {"U_t"+ss.str()+".csv"};
  		out<<fixed<<setprecision(4);			
  			for(int i=0; i<=M; ++i){
  				for(int j=0; j<=M; ++j){		
  					out<<U[i][j]<<" ";
  				}
  				out<<endl;
  			}
  		out.close();
	      }
  	}		
        if (rank == 0){	
  	   cout<<"iteration "<<t<<" done"<<endl;	
        }
  }

  double t_two = MPI_Wtime();
  double runtime = t_two - t_one;

  if (rank == 0){
     cout << "Runtime = " << runtime << endl;
  }

// now we need to delete the Ul array.
  for (int i = 0; i < 3; i++)
  {
      for (int j = 0; j < Jn; j++){
          delete[] Ul[i][j];
         }    
      delete[] Ul[i];
  }

  delete[] Ul;


// now we need to delete the U array.
  for (int i = 0; i < M+1; i++)
  {
      delete[] U[i];
  }

  delete[] U;


  MPI_Finalize();

//  return 0;

}
