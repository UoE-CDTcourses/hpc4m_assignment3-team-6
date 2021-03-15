//mpic++
//g++ -o w5_2 w5_2.cpp &&  ./w5_2


#include <iostream>
#include <fstream>
#include <mpi.h>
#include <math.h>
//#include "hdf5.h"
//using namespace hdf5.h;

//#include "H5Cpp.h"


//#define TRUE 1;
//using namespace hdf5;
#include <stdlib.h>
using namespace std;

class Vec2{
public:
  size_t x{},y{};
  void print() // defines a member function named print()
   {
       std::cout <<"V("<< x << ',' << y << ')' ;
   }
   Vec2 max(Vec2 o){

     return Vec2{std::max(x,o.x),std::max(y,o.y)};
   }
   Vec2 min(Vec2 o){
     return Vec2{std::min(x,o.x),std::min(y,o.y)};
   }
   Vec2 sub(Vec2 o){
     return Vec2{x-o.x,y-o.y};
   }

};
class Box{
public:
  Vec2 start{},end{};
  void print() // defines a member function named print()
   {
       std::cout <<"Box("<<start.x<<"-"<<end.x<<","<<start.y<<"-"<<end.y<<")"<<endl;
       //start.print();
       //end.print();

   }
   Box shrink(){
     return Box{Vec2{start.x+1,start.y+1},Vec2{end.x-1,end.y-1}};
  }
  Box intersect(Box o){
    return Box{start.max(o.start),end.min(o.end)};
  }
  bool non_empty(){
    return start.x<end.x && start.y<end.y;
  }
  size_t size(){
    return (end.x-start.x)*(end.y-start.y);
  }
  Box sub(Vec2 o){
    return Box{start.sub(o),end.sub(o)};
  }
};
class Source{
public:
  Box b;
  int source;
  int self_r;
  void print() // defines a member function named print()
   {
     b.print();
     cout<<"source "<<source<<"rank "<<self_r;

   }
 };
 double sq(double x){
   return x*x;
 }
 void fill(double *array,Box box, Vec2 tsz){


   size_t k=0;
   double tix=2./double(tsz.x-1);
   double tiy=2./double(tsz.y-1);
   for (size_t i=box.start.x;i<box.end.x;i++){
     for (size_t j=box.start.y;j<box.end.y;j++){
       //double(i)+double(j)*100.;//double(i)*tix+1.5*double(j)*tiy-2.5;//
       array[k]=exp(-40.*(sq(double(i)*tix-1.4)+sq(double(j)*tiy-1.)));
       /*if ((i^j)&1){
         array[k]+=0.01;
       }*/
       k++;
     }
   }


 }
//void fill(size_t )
void apply(double** array, size_t X, size_t Y,float del){
  //array [x*Y+y]
  for (size_t i=1;i<X-1;i++){
    for (size_t j=i*Y+1;j<(i+1)*Y-1;j++){
      array[2][j]=(-array[0][j]+2.*array[1][j])+del*(-4.*array[1][j]+array[1][j+1]+array[1][j-1]+array[1][j+Y]+array[1][j-Y]);
    }
  }

}
void sendbox(double* array, Box box, int to,size_t mY,bool sync){
  //box.print();
  //cout<<"to "<<to<<" "<<mY<<endl;
  double* val=&array[box.start.x*mY+box.start.y];
  size_t sz=box.size();
  bool mrows=true;//box.end.x>box.start.x+1;
  if (mrows){
    val=new double[sz];
    size_t k=0;
    for (size_t i=box.start.x;i<box.end.x;i++){
      for (size_t j=box.start.y+i*mY;j<box.end.y+i*mY;j++){
        val[k]=array[j];
        k++;
      }
    }
  }
  if (sync){
    MPI_Ssend(val,sz,MPI_DOUBLE,to,0,MPI_COMM_WORLD);
  }else{
    MPI_Bsend(val,sz,MPI_DOUBLE,to,0,MPI_COMM_WORLD);
  }

  if (mrows){
    delete[] val;
  }
}
void recvbox(double* array, Box box, int from,size_t mY){
  //box.print();
  //cout<<"fr "<<from<<" "<<mY<<endl;
  double* val=&array[box.start.x*mY+box.start.y];
  size_t sz=box.size();
  bool mrows=true;//box.end.x>box.start.x+1;
  if (mrows){
    val=new double[sz];
  }

  MPI_Recv(val,sz,MPI_DOUBLE,from,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
  if (mrows){
    size_t k=0;
    for (size_t i=box.start.x;i<box.end.x;i++){
      for (size_t j=box.start.y+i*mY;j<box.end.y+i*mY;j++){
        //cout<<"recv"<<array[j]<<" was "<<val[k]<<endl;
        array[j]=val[k];
        k++;
      }
    }

    delete[] val;
  }
}
void swapbox(double* array, Box* boxes,int size , int rank){
  Box self=boxes[rank];
  size_t mY=self.end.y-self.start.y;
  Box s=boxes[rank].shrink();
  Box ints;
  for (int i=0;i<size;i++){
    if (i==rank)i++;
    ints=s.intersect(boxes[i]);
    if (ints.non_empty()  ){
      //cout<<"###\n";
      //ints.print();
      sendbox(array,ints.sub(self.start),i,mY,false);
    }
  }
  s=boxes[rank];
  for (int i=0;i<size;i++){
    if (i==rank)i++;
    ints=s.intersect(boxes[i].shrink());
    if (ints.non_empty()  ){
      //cout<<"i="<<i<<" rank "<<rank<<endl;
      //ints.print();
      //cout<<"###\n";
      //ints.print();
      recvbox(array,ints.sub(self.start),i,mY);
    }
  }

}


void sksave(double* array, Box* boxes, int rank, int size, Vec2 tsz,ofstream *myfile){
Box big=Box{Vec2{},tsz};
Box me=boxes[rank];
if (rank!=0){
  sendbox(array,me.sub(me.start),0,me.end.y-me.start.y,true);
}
if (rank==0){
  double* biga=new double[tsz.x*tsz.y]{};


  size_t k=0;

  for (size_t i=me.start.x;i<me.end.x;i++){
    for (size_t j=me.start.y+i*tsz.y;j<me.end.y+i*tsz.y;j++){
      //cout<<"recv"<<array[j]<<" was "<<val[k]<<endl;
      biga[j]=array[k];
      k++;
    }
  }

  for (int i=1;i<size;i++){
    recvbox(biga, boxes[i], i,tsz.y);

  }
  //cout<<"myfile "<<myfile;
  (*myfile).write((char*)(biga),tsz.x*tsz.y*sizeof(double));
  delete[] biga;
}

}

int main(int argc, char *argv[]){
  int rank, size, ierr;
  MPI_Comm comm;

  comm  = MPI_COMM_WORLD;

  MPI_Init(NULL,NULL);
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);
  size_t sz=100;
  Vec2 tsz=Vec2{sz,sz};
  size_t X=tsz.x ,Y=tsz.y;
  double dx=2./double(X-1);
  double T=1.;
  double sqdxdt=0.45;//0.33;//0.5?

  size_t M=size_t(T/sqrt(sqdxdt)/dx)+1;//5*X
  int saveslices=10;
  int pM=M/saveslices+1;
  M=pM*saveslices;
  double dt=1./double(M);
  sqdxdt=sq(dt/dx);
  cout<<"X="<<X<<" M="<<M<<" dx= "<<dx<<" dt="<<dt<<" sqdxdt="<<sqdxdt<<endl;
  Box* boxes=new Box[size]{};
  if (argv[1][0]=='r'){
    cout<<"rows"<<endl;
    for (int i=0;i<size;i++){
      boxes[i].start=Vec2{0,(Y-2)*i/size};
      boxes[i].end=Vec2{X,(Y-2)*(i+1)/size+2};
      boxes[i].print();
    }
  }else if(argv[1][0]=='c'){
    cout<<"cols"<<endl;
    for (int i=0;i<size;i++){
      boxes[i].start=Vec2{(X-2)*i/size,0};
      boxes[i].end=Vec2{(X-2)*(i+1)/size+2,Y};
      boxes[i].print();
    }
  }else if(argv[1][0]=='b'){
  cout<<"boxs"<<endl;
  int s=int(sqrt(float(size)+0.001));
  size=s*s;
  cout<<"s="<<s<<" rank ="<<rank<<endl;
  if (rank>=s*s){MPI_Finalize(); return 0;}
  int k=0;
  for (int i=0;i<s;i++){
    for (int j=0;j<s;j++){
      boxes[k].start=Vec2{(X-2)*i/s,(Y-2)*j/s};
      boxes[k].end=Vec2{(X-2)*(i+1)/s+2,(Y-2)*(j+1)/s+2};
      boxes[k].print();
      k++;
    }
  }
}else{
    cout<<"Error, Invalid command \""<<argv[1]<<"\", please pass a valid command. To make this fun, I won't tell you what that is. You'll have to look at the source code. You do have the source code don't you? "<<endl;
    //argv[1];
    cout<< "This value is probably a 0, but if its a one it won't help you much. :"<<(argv[1][0]=='q')<<endl;

    MPI_Finalize();
    return 0;
  }

  //boxes[0].print();
  Box self=boxes[rank];
  Box small=self.shrink();

  //small.print();

  cout<<"rank "<<rank<<" of "<<size<<endl;
  size_t mX=self.end.x-self.start.x ,mY=self.end.y-self.start.y;
  size_t mSZ=mX*mY;

  //double (*array[3])[X*Y];//={new *double[X*Y],new *double[X*Y],new *double[X*Y]};
  double *array[3]={new double[mSZ]{},new double[mSZ]{},new double[mSZ]{}};
  //cout<<" "<<typeid(array).name()<<endl;
  //cout<<" "<<typeid(array[1]).name()<<endl;
  //cout<<" "<<array[1]<<endl;
  //cout<<" "<<array[1][2]<<endl;
  double *swap;
  fill(array[0],self,tsz);
  fill(array[1],self,tsz);
  fill(array[2],self,tsz);
  //bool wri=false;
  ofstream myfile;
  bool base=rank==0;
  char filenamestart[18]="w5_data_output__";
  filenamestart[15]=argv[1][0];
  //ilenamestart.append(argv[1]);
  //cout<<"filename"<<filenamestart;
  //MPI_Finalize();
  //return 0;
  if (base){

    myfile.open (filenamestart);//I tried to make this depend on mode, c++ strings are cursed.

  }
  //sksave(array[0], boxes, rank, size, tsz,&myfile);
  bool fine;
  double err;
  for (int i=0;i<M+1;i++){
    //myfile.write((char*)(array[0]),mSZ*sizeof(double));
    Vec2 gap=self.end.sub(self.start);
    apply(array ,gap.x,gap.y,sqdxdt);
    fine=true;
    err=0.;
    for (int k =0;k<self.size();k++){
      err+=abs( array[0][k]-array[2][k]);
    }
    //cout<<"h "<<err<<" "<<i<<endl;
    swapbox(array[2],  boxes,size , rank);
    err=0.;
    for (int k =0;k<self.size();k++){
      err+=abs( array[0][k]-array[2][k]);
    }
    //cout<<"j "<<err<<" "<<i<<endl;
    if ((i/pM)*pM==i){
      cout<<"i="<<i<<" M="<<M<<endl;
      sksave(array[0], boxes, rank, size, tsz,&myfile);
    }
    swap=array[0];
    array[0]=array[1];
    array[1]=array[2];
    array[2]=swap;

  }
  //write

  if (base){
    myfile.close();
  }
  //Vec2 a{2,6};
  //Source s{{1,2},{3,4},5,6};
  //a.print();
  MPI_Finalize();
  //for (int i=0;i<3;i++){

  for (int i=0;i<3;i++){
    delete[] array[i];
  }
  delete[] boxes;
}
