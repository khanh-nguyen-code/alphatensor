#include<cstdio>
#include<random>
#include"matmul.h"
#include"high_precision_timer.h"

const double eps = 1e-6;

void matmul(double *c, double *a, double *b, int dim[3]) {
    for (int i1=0; i1<dim[0]; i1++) {
        for (int i3=0; i3<dim[2]; i3++) {
            c[i1 * dim[2] + i3] = 0.0;
        }
    }
    for (int i1=0; i1<dim[0]; i1++) {
        for (int i3=0; i3<dim[2]; i3++) {
            for (int i2=0; i2 < dim[1]; i2++) {
                c[i1 * dim[2] + i3] += a[i1 * dim[1] + i2] * b[i2 * dim[2] + i3];
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
    auto matmul_opt = matmul_8_8_8;

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
    for (int i=0; i<5000000; i++)
    matmul(c, a, b, dim);
    auto t2 = timer::now();
    
    auto t3 = timer::now();
    for (int i=0; i<5000000; i++)
    matmul_opt(d, a, b);
    auto t4 = timer::now();

    std::printf("equal %d\n", is_equal(c, d, dim[0] * dim[2]));
    std::printf("unopt %d\n", t2-t1);
    std::printf("opt   %d\n", t4-t3);
}
