#include<iostream>
#include<random>
#include<openblas/cblas.h>
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

void matmul_blas(double *c, const double *a, const double *b, const int d0, const int d1, const int d2) {
    // c <- alpha a b + beta c
    cblas_dgemm(
        CblasRowMajor, CblasNoTrans, CblasNoTrans,
        d0, d2, d1,
        0.0, // alpha
        a, d0,
        b, d1,
        0.0, // beta
        c, d0
    );
}

const int step = 4;
auto matmul_unit = alphatensor::matmul_4_4_4;
void matmul_opt(double *c, const double *a, const double *b, const int d0, const int d1, const int d2) {

    if (d0 % step != 0 or d1 % step != 0 or d2 % step != 0) {
        std::cerr << "dim error (" << d0 << ", " << d1 << ", " << d2 << ")" << std::endl;
        std::exit(1);
    }
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
        for (i=0; i<d0; i += step) {
            for (j=0; j<d2; j += step) {
                for (k=0; k<d1; k += step) {
                    double a_sub[step * step];
                    double b_sub[step * step];
                    double c_sub[step * step];

                    for (int s_i=0; s_i < step; s_i++) {
                        for (int s_j=0; s_j < step; s_j++) {
                            a_sub[s_i * step + s_j] = a[(i + s_i) * d1 + (k + s_j)];
                            b_sub[s_i * step + s_j] = b[(k + s_i) * d2 + (j + s_j)];
                        }
                    }
                    matmul_unit(c_sub, a_sub, b_sub);
                    for (int s_i=0; s_i < step; s_i++) {
                        for (int s_j=0; s_j < step; s_j++) {
                            c[(i + s_i) * d2 + (j + s_j)] += c_sub[s_i * step + s_j];
                        }
                    }
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
    std::default_random_engine engine;
    std::uniform_real_distribution<double> dist(0, 1);
    
    const int n = 1600;
    int dim[3] = {n*step, n*step, n*step};

    double* a = new double[dim[0] * dim[1]];
    double* b = new double[dim[1] * dim[2]];
    double* c = new double[dim[0] * dim[2]];
    double* d = new double[dim[0] * dim[2]];
    double* e = new double[dim[0] * dim[2]];


    for (int i=0; i < dim[0] * dim[1]; i++) {
        a[i] = dist(engine);
    }
    for (int i=0; i < dim[1] * dim[2]; i++) {
        b[i] = dist(engine);
    }
    for (int i=0; i < dim[0] * dim[2]; i++) {
        c[i] = 0;
        d[i] = 0;
        e[i] = 0;
    }
    auto t1 = timer::now();
    matmul(c, a, b, dim[0], dim[1], dim[2]);
    auto t2 = timer::now();
    
    auto t3 = timer::now();
    matmul_blas(d, a, b, dim[0], dim[1], dim[2]);
    auto t4 = timer::now();
    
    auto t5 = timer::now();
    matmul_opt(e, a, b, dim[0], dim[1], dim[2]);
    auto t6 = timer::now();

    std::printf("equal %d\n", is_equal(c, d, dim[0] * dim[2]) + is_equal(c, e, dim[0] * dim[2]));
    std::printf("blas           %f\n", ((double) (t4-t3)) / ((double) (t2-t1)));
    std::printf("alphatensor    %f\n", ((double) (t6-t5)) / ((double) (t2-t1)));
}
