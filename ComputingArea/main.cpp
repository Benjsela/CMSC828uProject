#include<stdio.h>
#include<math.h>







int main(){
  long double radius = 5.0;
  long double exponent = 28.0;
  long double radmult = powl(radius,exponent);
  printf("radmult = %Le\n",radmult);
  double result = tgamma(15);
  double pi = M_PI;
  double numerator = std::pow(pi,14);
  printf("numerator = %f\n",numerator);
  printf("pi = %f\n",pi);
  printf("result = %f\n",result);
  double ans = radmult*numerator/result;
  printf("ans = %f\n",ans);
  return 0;
  


}


