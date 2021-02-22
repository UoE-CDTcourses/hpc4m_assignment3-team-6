#include <iostream>
#include <cmath>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <mpi.h>
#include <chrono>

using namespace std;

int main(int argc, char* argv[]){
  
  int rank, size, ierr;
  MPI_Comm comm;

  comm  = MPI_COMM_WORLD;      //store the global communicator in comm variable

  MPI_Init(NULL,NULL);         //initialise MPI
  MPI_Comm_rank(comm, &rank);  //process identifier stored in rank variable
  MPI_Comm_size(comm, &size);  //number of processes stored in size variable
  
  double start, end;
  double writingT, writingStart, writingEnd;
  writingT = 0;
  

  int M = 2305;  // M length intervals ==> M+1 = 2306 grid points and rows
  double T = 1;  // final time.
  double dt = 0.2/M;
  int N = T/dt;
  int t1=0.333/dt, t2=0.666/dt, t3=N; // points at which we want to print our results.
  double dy = 2./M;
  double dy2 = dy*dy;
  double dtdy2 = dt*dt/dy2;
  
  MPI_Barrier(comm);
  start = MPI_Wtime();      // start timing here
  
  int J;
  //first compute how many rows per process
  if ((M+1-2)%size==0) {
    J = int((M+1-2)/size) + 2;
    if (rank==0) {
      cout << "Using " << J << " rows per process" << endl;
      cout << "Number of Time Steps: " << N << endl;
    }
  } else {
    if (rank==0) {
      cout << "Bad match for M and size - (M+1-2)/size = " << (M+1-2)/size << endl;
    }
    
  }

  // dynamically allocate enough memory to create the array.
  // !! ensure that the memory is allocated in all 1 block !!
  
  double*** U = new double**[3];
  double**  Ub = new double*[3*J];
  double*   Uc = new double[3*J*(M+1)];
  for (int i=0; i<3; i++) {
      for (int j=0; j<J; j++) {
          Ub[J*i+j] = Uc + (J*i+j)*(M+1);
      }
      U[i] = Ub + J*i;
  }
  
  // dynamically allocate enough memory to store the array on the root process.
  // !! ensure that the memory is allocated in all 1 block !!
  
  double** Usol = new double*[M+1];
  double*  Usolb = new double[(M+1)*(M+1)];
  for (int i=0; i<(M+1); i++) {
      Usol[i] = Usolb + (M+1)*i;
  }
  
  // apart from array deletion at the end, everything else stays as usual..

  // initialize numerical array with given conditions on each process.
  int yi;
  for (int i=0; i<J; ++i){        // i is y index   (J rows on each process)
    for (int j=0; j<M+1; ++j){    // j is x index
      yi = rank*(J-2)+i;
      U[0][i][j] = U[1][i][j] = exp( -40 * ( (yi*dy-1-0.4)*(yi*dy-1-0.4) + (j*dy-1)*(j*dy-1) ) );
    }
  }
  
  //special boundary conditions (process specific for first and last process)
  for (int i=0; i<J; i++) {
    U[0][i][0] = U[0][i][M] = U[1][i][0] = U[1][i][M] = U[2][i][0] = U[2][i][M] = 0;
  }

  if (rank==0) {
    for (int i=0; i<=M; ++i){
      U[0][0][i] = U[1][0][i] = U[2][0][i] = 0;
    }
  } else if (rank == size-1) {
    for (int i=0; i<=M; ++i){
      U[0][J-1][i] = U[1][J-1][i] = U[2][J-1][i] = 0;
    }
  }
  
  // NOW DOING GATHERING FOR FIRST OUTPUT
  
  MPI_Barrier(comm);
  writingStart = MPI_Wtime();
  
  MPI_Gather(&(U[0][0][0]), (J-2)*(M+1), MPI_DOUBLE, &(Usol[0][0]), (J-2)*(M+1), MPI_DOUBLE, 0, comm);
  
  if (rank==size-1) {
    
    MPI_Ssend(&(U[0][J-2][0]), 2*(M+1), MPI_DOUBLE, 0, 0, comm);
    
  }
  if (rank==0) {
    MPI_Recv(&(Usol[M-1][0]), 2*(M+1), MPI_DOUBLE, size-1, 0, comm, MPI_STATUS_IGNORE);
  }
  
  
  if (rank==0) {
    
    //Print to file
    ofstream out; out.open("U_t0.csv");
    out<<fixed<<setprecision(4);
    for(int i=0; i<=M; ++i){
      for(int j=0; j<=M; ++j){
        out<<Usol[i][j]<<" ";
      }
      out<<endl;
    }
    
    out.close();
    
  }
  
  writingEnd = MPI_Wtime();
  writingT += (writingEnd-writingStart);      // keep track of how much time is spent writing
  

  // use numerical scheme to obtain the future values of U.
  for (int t=1; t<=N; ++t){
    
    for (int i=1; i<J-1; ++i){
      for (int j=1; j<M; ++j){
        U[2][i][j] = 2*U[1][i][j] - U[0][i][j]
                 + dtdy2*( U[1][i+1][j] + U[1][i-1][j] + U[1][i][j+1] + U[1][i][j-1] - 4*U[1][i][j] );
      }
    }

    
    //SWAP TIME   -    Split into even and odd processes
    
    if (rank%2==0) {
      if (rank!=size-1) {
        MPI_Ssend(&(U[2][J-2][0]), M+1, MPI_DOUBLE, rank+1, 2*(rank)+1, comm);   //exchange all the second last rows first
      }
      
      if (rank!=0){
        MPI_Recv(&(U[2][0][0]), M+1, MPI_DOUBLE, rank-1, 2*(rank-1)+1, comm, MPI_STATUS_IGNORE);
        MPI_Ssend(&(U[2][1][0]), M+1, MPI_DOUBLE, rank-1, 2*(rank), comm);   //now exchange all the second rows
      }
      
      if (rank!=size-1) {
        MPI_Recv(&(U[2][J-1][0]), M+1, MPI_DOUBLE, rank+1, 2*(rank+1), comm, MPI_STATUS_IGNORE);
      }
      
    } else {
      
      MPI_Recv(&(U[2][0][0]), M+1, MPI_DOUBLE, rank-1, 2*(rank-1)+1, comm, MPI_STATUS_IGNORE);
      
      if (rank!=size-1) {
        MPI_Ssend(&(U[2][J-2][0]), M+1, MPI_DOUBLE, rank+1, 2*(rank)+1, comm);   //exchange all the second last rows first
        MPI_Recv(&(U[2][J-1][0]), M+1, MPI_DOUBLE, rank+1, 2*(rank+1), comm, MPI_STATUS_IGNORE);
      }
      
      MPI_Ssend(&(U[2][1][0]), M+1, MPI_DOUBLE, rank-1, 2*(rank), comm);   //now exchange all the second rows
      
    }
    
    // update the previous times.
    for (int i=0; i<J; ++i){
      for (int j=0; j<M+1; ++j){
        U[0][i][j] = U[1][i][j];
        U[1][i][j] = U[2][i][j];
        }
    }
    
     // print out files for fixed times
    if(t==t1 || t==t2 || t==t3){
      
      MPI_Barrier(comm);
      writingStart = MPI_Wtime();
      
      stringstream ss;
      ss << fixed << setprecision(2) << t*dt; // this ensures that the double value gets converted
      string time = ss.str();						 // to string with only 2 trailing digits.
      
      //Gather the final solution onto the root process
      MPI_Gather(&(U[2][0][0]), (J-2)*(M+1), MPI_DOUBLE, &(Usol[0][0]), (J-2)*(M+1), MPI_DOUBLE, 0, comm);
      if (rank==size-1) {
        MPI_Ssend(&(U[2][J-2][0]), 2*(M+1), MPI_DOUBLE, 0, 0, comm);
      }
      
      if (rank==0) {
        MPI_Recv(&(Usol[M-1][0]), 2*(M+1), MPI_DOUBLE, size-1, 0, comm, MPI_STATUS_IGNORE);
      }
      
      if (rank==0) {
        
        //Print to file
        ofstream out; out.open("U_t"+ss.str()+".csv");
        out<<fixed<<setprecision(4);
          for(int i=0; i<=M; ++i){
            for(int j=0; j<=M; ++j){
              out<<Usol[i][j]<<" ";
            }
            out<<endl;
          }
        
        out.close();
        
      }
      
      writingEnd = MPI_Wtime();
      writingT += (writingEnd-writingStart);
      

    }
    if (rank==0) {
      cout<<"iteration "<<t<<" done"<<endl;
    }
    
  }

  // now we need to delete the array.
  
  delete[] U; delete[] Ub; delete[] Uc;
  delete[] Usol; delete[] Usolb;
  
  MPI_Barrier(comm);             // find time elapsed
  end = MPI_Wtime();
  if (rank==0) {
    cout << "Time taken for " << size << " processes: " << end-start << " seconds" << endl;
    cout << "Time taken for writing the data: " << writingT << " seconds" << endl;
  }
  
  
  MPI_Finalize();      //need to finalise MPI


  return 0;

}
