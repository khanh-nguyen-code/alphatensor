import numpy as np

DEBUG = True


def realign_index(d1: int, d2: int) -> np.ndarray:
    """
    realign column major to row major
    """
    c2r = np.empty(shape=(d1 * d2,), dtype=np.int64)
    for i1 in range(d1):
        for i2 in range(d2):
            r_i = i1 * d2 + i2
            c_i = i2 * d1 + i1

            c2r[r_i] = c_i
    return c2r


def to_c_code(key: str, u: np.ndarray, v: np.ndarray, w: np.ndarray) -> tuple[str, str]:
    def make_coef(c: int, v: str) -> str:
        if c >= 0:
            return "".join([f" + {v}" for _ in range(abs(c))])
        else:
            return "".join([f" - {v}" for _ in range(abs(c))])

    rank = u.shape[1]
    lines = []
    for r in range(rank):
        a_sum = ""
        for i in range(u.shape[0]):
            a_sum += make_coef(u[i, r], f"a[{i}]")
        b_sum = ""
        for i in range(v.shape[0]):
            b_sum += make_coef(v[i, r], f"b[{i}]")

        line = f"double m_{r} = ({a_sum}) * ({b_sum});"
        lines.append(line)

    for i in range(w.shape[0]):
        m_sum = ""
        for r in range(rank):
            m_sum += make_coef(w[i, r], f"m_{r}")

        line = f"c[{i}] = {m_sum};"
        lines.append(line)

    lines = "\n".join(lines)

    d1, d2, d3 = key.split(",")
    header = f"void matmul_{d1}_{d2}_{d3}(double* c, double* a, double* b);"
    code = f"""
void matmul_{d1}_{d2}_{d3}(double* c, double* a, double* b) {{
{lines}
}}
    """
    return header, code


if __name__ == "__main__":
    in_file = "factorizations_r.npz"
    out = "matmul"
    debug_key = "4,4,4"

    with open(in_file, "rb") as f:
        factorizations = dict(np.load(f, allow_pickle=True))
    # ALIGN
    aligned_factorizations = {}
    for key, (u, v, w) in factorizations.items():
        # realign index
        d1, d2, d3 = key.split(",")
        d1, d2, d3 = int(d1), int(d2), int(d3)
        c2r = realign_index(d1, d3)
        w = w[c2r, :]
        aligned_factorizations[key] = (u, v, w)

    factorizations = aligned_factorizations

    # PRINT
    for key, (u, v, w) in factorizations.items():
        d1, d2, d3 = key.split(",")
        d1, d2, d3 = int(d1), int(d2), int(d3)
        rank = u.shape[1]
        print(key, rank, rank / (d1 * d2 * d3))
    # GEN C CODE
    header_list = []
    code_list = []
    for key, (u, v, w) in aligned_factorizations.items():
        # convert
        rank = u.shape[1]
        if DEBUG and key != debug_key:
            continue

        header, code = to_c_code(key, u, v, w)
        header_list.append(header)
        code_list.append(code)

    header_list = "\n".join(header_list)

    header = f"""
#ifndef __MATMUL__
#define __MATMUL__
namespace alphatensor {{
{header_list}
}};
#endif
    """

    code_list = "\n".join(code_list)

    code = f"""
#include"{out}.h"
namespace alphatensor {{
{code_list}
}}
    """

    with open(out + ".h", "w") as f:
        f.write(header)
    with open(out + ".cpp", "w") as f:
        f.write(code)
