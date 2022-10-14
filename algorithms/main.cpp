#include<cstdio>
#include<random>
#include"matmul.h"
#include"high_precision_timer.h"

const double eps = 1e-6;

void matmul(double *c, const double *a, const double *b, const int d0, const int d1, const int d2) {
    {
        int i, j;
        #pragma omp parallel for shared(c) private(i, j)
        for (i=0; i<d0; i++) {
            for (j=0; j<d2; j++) {
                c[i * d2 + j] = 0.0;
            }
        }
    }
    {
        int i, j, k;
        #pragma omp parallel for shared(c) private(i, j, k)
        for (i=0; i<d0; i++) {
            for (j=0; j<d2; j++) {
                for (k=0; k<d1; k++) {
                    c[i * d2 + j] += a[i * d1 + k] * b[k * d2 + j];
                }
            }
        }
    }
}

void print_arr(double *a, int d1, int d2) {
    for (int i1=0; i1<d1; i1++) {
        for (int i2=0; i2<d2; i2++) {
            std::printf("%f\t", a[i1 * d2 + i2]);
        }
        std::printf("\n");
    }
    std::printf("\n");
}
bool is_equal(double *a, double *b, int d) {
    for (int i=0; i<d; i++) {
        double diff = a[i] - b[i];
        diff = (diff >= 0) ? diff : -diff;
        if (diff > eps) {
            return false;
        }
    }
    return true;
}

int main() {
    std::default_random_engine e;
    std::uniform_real_distribution<double> dist(0, 1);
    
    int dim[3] = {8, 8, 8};
    auto matmul_opt = alphatensor::matmul_8_8_8;

    double* a = new double[dim[0] * dim[1]];
    double* b = new double[dim[1] * dim[2]];
    double* c = new double[dim[0] * dim[2]];
    double* d = new double[dim[0] * dim[2]];

    for (int i=0; i < dim[0] * dim[1]; i++) {
        a[i] = dist(e);
    }
    for (int i=0; i < dim[1] * dim[2]; i++) {
        b[i] = dist(e);
    }
    for (int i=0; i < dim[0] * dim[2]; i++) {
        c[i] = 0;
        d[i] = 0;
    }
    auto t1 = timer::now();
    matmul(c, a, b, dim[0], dim[1], dim[2]);
    auto t2 = timer::now();
    
    auto t3 = timer::now();
    matmul_opt(d, a, b);
    auto t4 = timer::now();

    std::printf("equal %d\n", is_equal(c, d, dim[0] * dim[2]));
    std::printf("unopt %d\n", t2-t1);
    std::printf("opt   %d\n", t4-t3);
}
