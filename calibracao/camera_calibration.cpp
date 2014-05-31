#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>

using namespace std;

///////////////////////////////////////////////
///////                                 ///////
///////         CLASS DEFINITION        ///////
///////                                 ///////
///////////////////////////////////////////////

class Camera{
    public:
        cv::Mat image;

        void Initialize();
        void ReadPointsList(char* filename);
        void Run();
        static void MouseCallback( int event, int x, int y, int flags, void* param);

        bool using3DPoints;

    private:
        cv::Mat homographic;
        cv::Mat cameraMatrix;

        bool readingPoints;

        vector<cv::Point2d> points2D; 
        vector<cv::Point2d> pixels2D; 

        vector<cv::Point3d> points3D; 
        vector<cv::Point2d> pixels3D; 

        fstream PointsList;
        void SavePointsList();

        void ComputeHomographicMatrix();
        void Read2DPoint(char c);
        void Estimate2DPoint();        
        void Print2DPoints(); 

        void ComputeCameraMatrix();
        void Read3DPoint(char c);
        void Estimate3DPoint();        
        void Print3DPoints();        
};

//////////////////////////////////////
///////                        ///////
///////         PUBLIC         ///////
///////                        ///////
//////////////////////////////////////

void Camera::Initialize()
{
    readingPoints = false;

    cv::namedWindow( "Image", CV_WINDOW_AUTOSIZE );

    cv::setMouseCallback( "Image", MouseCallback, (void*)this);

    PointsList.open("output.txt", std::fstream::out);

    homographic  = cv::Mat::zeros(3,3,CV_64F);
    cameraMatrix = cv::Mat::zeros(3,4,CV_64F);
}

void Camera::Run()
{
    while(true)
    {
        cv::imshow("Image", image);

        int c = cv::waitKey();

        if(readingPoints){
            if(using3DPoints){
                Read3DPoint(c);
                Estimate3DPoint();
            }else{
                Read2DPoint(c);
                Estimate2DPoint();
            }
        }

        if( (c & 255) == 27 ) {
            SavePointsList();
            cout << "Exiting ...\n";
            break;
        }

        switch( (char)c ){
            case 'p':
                if(using3DPoints)
                    Print3DPoints();
                else
                    Print2DPoints();
                break;
            case 'c':
                if(using3DPoints)
                    ComputeCameraMatrix();
                else
                    ComputeHomographicMatrix();
                break;
            case 'm':
                using3DPoints = !using3DPoints;
                if(using3DPoints)
                    cout << "Using 3D points" << endl;
                else
                    cout << "Using 2D points" << endl;
                break;
        }
    }
}

void Camera::ReadPointsList(char* filename)
{
    fstream input;
    input.open(filename, std::fstream::in);

    int N;
    input >> N;
    cout << N << endl;
    if(using3DPoints){
        points3D.resize(N);
        pixels3D.resize(N);
        for(int i=0;i<N;i++)
            input >> pixels3D[i].x >> pixels3D[i].y >> points3D[i].x >> points3D[i].y >> points3D[i].z;
        ComputeCameraMatrix();
    }else{
        points2D.resize(N);
        pixels2D.resize(N);
        for(int i=0;i<N;i++)
            input >> pixels2D[i].x >> pixels2D[i].y >> points2D[i].x >> points2D[i].y;
        ComputeHomographicMatrix();
    }
}

void Camera::MouseCallback( int event, int x, int y, int flags, void* param)
{
    Camera* cam = (Camera*) param;

    if( event == cv::EVENT_LBUTTONDOWN ){

        if(cam->using3DPoints){
            cam->pixels3D.push_back(cv::Point2d(x,y));
            cout << "Click at " << cam->pixels3D[cam->pixels3D.size()-1].x << ' ' 
                                << cam->pixels3D[cam->pixels3D.size()-1].y << endl;
        }else{
            cam->pixels2D.push_back(cv::Point2d(x,y));
            cout << "Click at " << cam->pixels2D[cam->pixels2D.size()-1].x << ' ' 
                                << cam->pixels2D[cam->pixels2D.size()-1].y << endl;
        }
        cam->readingPoints = true;
    }
}

///////////////////////////////////////
///////                         ///////
///////         PRIVATE         ///////
///////                         ///////
///////////////////////////////////////

void Camera::SavePointsList()
{
    if(using3DPoints){
        int N=pixels3D.size();
        PointsList << N << endl;
        for(int i=0;i<N;i++){
            PointsList << pixels3D[i].x << ' ' << pixels3D[i].y << ' ';
            PointsList << points3D[i].x << ' ' << points3D[i].y << ' ' << points3D[i].z << endl;
        }
    }else{
        int N=pixels2D.size();
        PointsList << N << endl;
        for(int i=0;i<N;i++){
            PointsList << pixels2D[i].x << ' ' << pixels2D[i].y << ' ';
            PointsList << points2D[i].x << ' ' << points2D[i].y << endl;
        }
    }
}

//////////////////////////////////
///////         2D         ///////
//////////////////////////////////

void Camera::Read2DPoint(char c1)
{
    char input[20];
    int head=0;
    bool readingFirst=true;


    int c=c1;

    while(true){

        input[head]=c%256;
        input[head+1]='\0';

        cout << input[head];cout.flush();

        if(input[head]==' '){
            input[head]='\0';
            if(readingFirst){
                readingFirst=false;
                points2D.push_back(cv::Point2d());
                points2D[points2D.size()-1].x=atof(input);
                head=-1;
            }else{
                readingPoints=false;
                points2D[points2D.size()-1].y=atof(input);
                cout << "Added point (" << points2D[points2D.size()-1].x << ',' << points2D[points2D.size()-1].y << ")" << endl;
                break;
            }
        }

        head++; 

        c = cv::waitKey();
    }
}

void Camera::Estimate2DPoint()
{
    cv::Mat p(3,1,CV_64F);
    p.at<double>(0,0) = points2D[points2D.size()-1].x;
    p.at<double>(1,0) = points2D[points2D.size()-1].y;
    p.at<double>(2,0) = 1.0;

    cv::Mat s(3,1,CV_64F);
    s=homographic*p;
    cout << "Estimated pixel: " << s/s.at<double>(2,0) << endl;

    p.at<double>(0,0) = pixels2D[pixels2D.size()-1].x;
    p.at<double>(1,0) = pixels2D[pixels2D.size()-1].y;
    p.at<double>(2,0) = 1.0;

    s=homographic.inv()*p;
    cout << "Estimated point: " << s/s.at<double>(2,0) << endl;
}

void Camera::Print2DPoints()
{
    cout << "Pixels: ";
    for(int i=0; i<pixels2D.size(); i++)
        cout << "(" << pixels2D[i].x << ',' << pixels2D[i].y << ") ";
    cout << endl;
    cout << "Points: ";
    for(int i=0; i<points2D.size(); i++)
        cout << "(" << points2D[i].x << ',' << points2D[i].y << ") ";
    cout << endl;
}

void Camera::ComputeHomographicMatrix()
{
    int numPoints = points2D.size();

    if(numPoints<4)
        return;

    int lines = 2*numPoints;

    // Initialize lines of matrix A
    double (*tempA)[8] = new double[lines][8];
    for(int p=0; p<numPoints; p++){
        double tempX[8] = {points2D[p].x, points2D[p].y, 1, 0, 0, 0, -pixels2D[p].x*points2D[p].x, -pixels2D[p].x*points2D[p].y};
        memcpy(tempA[2*p],tempX,8*sizeof(double));
        double tempY[8] = {0, 0, 0, points2D[p].x, points2D[p].y, 1, -pixels2D[p].y*points2D[p].x, -pixels2D[p].y*points2D[p].y};
        memcpy(tempA[2*p+1],tempY,8*sizeof(double));
    }

    cv::Mat A(lines,8,CV_64F, tempA);

    // Print matrix A
    // cout << "A = "<< endl << " " << A << endl << endl;

    // Initialize lines of matrix b
    double *tempB = new double[lines];
    for(int p=0; p<numPoints; p++){
        tempB[2*p]   = pixels2D[p].x;
        tempB[2*p+1] = pixels2D[p].y;
    }

    cv::Mat b(lines,1,CV_64F,tempB);

    // Print matrix b
    // cout << "b = "<< endl << " " << b << endl << endl;

    // Compute solution
    cv::Mat solution = ((A.t()*A).inv())*(A.t()*b);
    solution.push_back(1.0);

    // Copy solution to homographic matrix
    for (int i=0; i<3; i++)
        for (int j=0; j<3; j++)
            homographic.at<double>(i,j) = solution.at<double>(3*i+j,0);

    // Print homographic matrix
     cout << "Homographic:" << homographic << endl;
}


//////////////////////////////////
///////         3D         ///////
//////////////////////////////////

void Camera::Read3DPoint(char c1)
{
    char input[20];
    int head=0;
    bool readingFirst=true;
    bool readingSecond=false;

    int c=c1;

    while(true){

        input[head]=c%256;
        input[head+1]='\0';

        cout << input[head];cout.flush();

        if(input[head]==' '){
            input[head]='\0';
            if(readingFirst){
                readingFirst=false;
                readingSecond=true;
                points3D.push_back(cv::Point3d());
                points3D[points3D.size()-1].x=atof(input);
                head=-1;
            }else if(readingSecond){
                readingSecond=false;
                points3D[points3D.size()-1].y=atof(input);
                head=-1;
            }else{
                readingPoints=false;
                points3D[points3D.size()-1].z=atof(input);
                cout << "Added point (" << points3D[points3D.size()-1].x 
                                 << ',' << points3D[points3D.size()-1].y 
                                 << ',' << points3D[points3D.size()-1].z <<")" << endl;
                break;
            }
        }

        head++; 

        c = cv::waitKey();
    }
}

void Camera::Estimate3DPoint()
{
    cv::Mat p(4,1,CV_64F);
    p.at<double>(0,0) = points3D[points3D.size()-1].x;
    p.at<double>(1,0) = points3D[points3D.size()-1].y;
    p.at<double>(2,0) = points3D[points3D.size()-1].z;
    p.at<double>(3,0) = 1.0;

    cv::Mat s(3,1,CV_64F);
    s=cameraMatrix*p;
    cout << "Estimated pixel: " << s/s.at<double>(2,0) << endl;

//    p.at<double>(0,0) = pixels3D[pixels3D.size()-1].x;
//    p.at<double>(1,0) = pixels3D[pixels3D.size()-1].y;
//    p.at<double>(2,0) = pixels3D[pixels3D.size()-1].z;
//    p.at<double>(3,0) = 1.0;
//    p=cameraMatrix.inv()*p;
//    cout << "Estimated point: " << p/p.at<double>(3,0) << endl;
}

void Camera::Print3DPoints()
{
    cout << "Pixels: ";
    for(int i=0; i<pixels3D.size(); i++)
        cout << "(" << pixels3D[i].x << ',' << pixels3D[i].y << ") ";
    cout << endl;
    cout << "Points: ";
    for(int i=0; i<points3D.size(); i++)
        cout << "(" << points3D[i].x << ',' << points3D[i].y << ',' << points3D[i].z << ") ";
    cout << endl;
}

void Camera::ComputeCameraMatrix()
{
    int numPoints = points3D.size();

    if(numPoints<6)
        return;

    int lines = 2*numPoints;

    // Initialize lines of matrix A
    double (*tempA)[11] = new double[lines][11];
    for(int p=0; p<numPoints; p++){
        double tempX[11] = {points3D[p].x, points3D[p].y, points3D[p].z, 1, 0, 0, 0, 0, -pixels3D[p].x*points3D[p].x, -pixels3D[p].x*points3D[p].y, -pixels3D[p].x*points3D[p].z};
        memcpy(tempA[2*p],tempX,11*sizeof(double));
        double tempY[11] = {0, 0, 0, 0, points3D[p].x, points3D[p].y, points3D[p].z, 1, -pixels3D[p].y*points3D[p].x, -pixels3D[p].y*points3D[p].y, -pixels3D[p].y*points3D[p].z};
        memcpy(tempA[2*p+1],tempY,11*sizeof(double));
    }

    cv::Mat A(lines,11,CV_64F, tempA);

    // Print matrix A
    // cout << "A = "<< endl << " " << A << endl << endl;

    // Initialize lines of matrix b
    double *tempB = new double[lines];
    for(int p=0; p<numPoints; p++){
        tempB[2*p]   = pixels3D[p].x;
        tempB[2*p+1] = pixels3D[p].y;
    }

    cv::Mat b(lines,1,CV_64F,tempB);

    // Print matrix b
    // cout << "b = "<< endl << " " << b << endl << endl;

    // Compute solution
    cv::Mat solution = ((A.t()*A).inv())*(A.t()*b);
    solution.push_back(1.0);

    // Copy solution to camera matrix
    for (int i=0; i<3; i++)
        for (int j=0; j<4; j++)
            cameraMatrix.at<double>(i,j) = solution.at<double>(4*i+j,0);

    // Print camera matrix
     cout << "CameraMatrix:" << cameraMatrix << endl;


    cv::Mat cam,rot,trans;
    decomposeProjectionMatrix(cameraMatrix,cam,rot,trans);

    cout << "Intrinsicos " << cam << endl;
    cout << "Rot " << rot << endl;
    cout << "Trans " << trans << endl;
}

////////////////////////////////////
///////                      ///////
///////         MAIN         ///////
///////                      ///////
////////////////////////////////////

int main( int argc, char** argv )
{
    Camera cam;

    cam.image = cv::imread( argv[1], 1 );

    if( argc < 2 || !cam.image.data )
    {
        cout << "No image data" << endl;
        return -1;
    }else{
        cout << cam.image.size().width << 'x' << cam.image.size().height << endl;
    }


    if(argv[1][5]=='3'){ // imgs\3 <--- 3d
        cam.using3DPoints = true;
        cout << "Using 3D points" << endl;
    }else{ // 2d
        cam.using3DPoints = false;
        cout << "Using 2D points" << endl;
    }

    cam.using3DPoints = true;

    cam.Initialize();

    if(argc == 3)
        cam.ReadPointsList(argv[2]);

    cam.Run();

    return 0;
}

