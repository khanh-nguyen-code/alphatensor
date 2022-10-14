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


def optimize(u: np.ndarray, v: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sub_j = []
    for j in range(u.shape[1]):
        if np.sum(u[:, j] != 0) > 0 and np.sum(v[:, j] != 0) > 0:
            sub_j.append(j)
    u, v, w = u[:, sub_j], v[:, sub_j], w[:, sub_j]
    return u, v, w


def convert_8_8_10_to_8_8_8(u: np.ndarray, v: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sub_i = []
    for i1 in range(8):
        for i2 in range(8):
            sub_i.append(i1 * 10 + i2)
    v = v[sub_i, :]
    w = w[sub_i, :]

    return optimize(u, v, w)


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


def to_cl_code(key: str, u: np.ndarray, v: np.ndarray, w: np.ndarray) -> str:
    def make_coef(c: int, v: str) -> str:
        if c >= 0:
            return "".join([f" + {v}" for _ in range(abs(c))])
        else:
            return "".join([f" - {v}" for _ in range(abs(c))])

    d1, d2, d3 = key.split(",")
    d1, d2, d3 = int(d1), int(d2), int(d3)
    rank = u.shape[1]
    lines = []
    for i in range(d1):
        for j in range(d2):
            line = f"double a_{i * d2 + j} = a_arr[({i} + a_i) * a_i_d + ({j} + a_j)];"
            lines.append(line)

    for i in range(d2):
        for j in range(d3):
            line = f"double b_{i * d3 + j} = b_arr[({i} + b_i) * b_i_d + ({j} + b_j)];"
            lines.append(line)

    for r in range(rank):
        a_sum = ""
        for i in range(u.shape[0]):
            a_sum += make_coef(u[i, r], f"a_{i}")
        b_sum = ""
        for i in range(v.shape[0]):
            b_sum += make_coef(v[i, r], f"b_{i}")

        line = f"double m_{r} = ({a_sum}) * ({b_sum});"
        lines.append(line)

    for i in range(w.shape[0]):
        m_sum = ""
        for r in range(rank):
            m_sum += make_coef(w[i, r], f"m_{r}")

        line = f"c_{i} = {m_sum};"
        lines.append(line)

    for i in range(d1):
        for j in range(d3):
            line = f"c_arr[({i} + c_i) * c_i_d + ({j} + c_j)] = c_{i * d3 + j};"
            lines.append(line)

    lines = "\n".join(lines)

    code = f"""
void matmul_{d1}_{d2}_{d3}(
    double* c_arr, const double* a_arr, const double* b_arr,
    const int c_i, const int c_j, const int a_i, const int a_j, const int a_i, const int a_j,
    const int c_i_d, const int c_j_d, const int a_i_d, const int a_j_d, const int a_i_d, const int a_j_d,
) {{
{lines}
}}
    """
    return code


if __name__ == "__main__":
    in_file = "factorizations_r.npz"
    out = "matmul"
    debug_key = "2,2,2"

    with open(in_file, "rb") as f:
        factorizations = dict(np.load(f, allow_pickle=True))

    # ALIGN
    aligned_factorizations = {}
    for key, (u, v, w) in factorizations.items():
        # realign index
        a, b, c = key.split(",")
        c2r = realign_index(int(a), int(c))
        w = w[c2r, :]
        aligned_factorizations[key] = (u, v, w)

    factorizations = aligned_factorizations

    # 8 8 10 to 8 8 8
    u, v, w = factorizations["8,8,10"]
    factorizations["8,8,8"] = convert_8_8_10_to_8_8_8(u, v, w)

    # OPTIMIZE
    for key, (u, v, w) in factorizations.items():
        u, v, w = optimize(u, v, w)
        factorizations[key] = u, v, w

    # GEN C CODE
    header_list = []
    code_list = []
    for key, (u, v, w) in aligned_factorizations.items():
        # convert
        rank = u.shape[1]
        print(key, rank)
        if DEBUG and key != debug_key:
            continue

        header, code = to_c_code(key, u, v, w)
        header_list.append(header)
        code_list.append(code)

    header_list = "\n".join(header_list)

    header = f"""
#ifndef __MATMUL__
#define __MATMUL__

{header_list}

#endif
    """

    code_list = "\n".join(code_list)

    code = f"""
#include"{out}.h"
{code_list}
    """

    with open(out + ".h", "w") as f:
        f.write(header)
    with open(out + ".cpp", "w") as f:
        f.write(code)

    # GEN CL CODE
    code_list = []
    for key, (u, v, w) in aligned_factorizations.items():
        # convert
        rank = u.shape[1]
        print(key, rank)
        if DEBUG and key != debug_key:
            continue

        code = to_cl_code(key, u, v, w)
        code_list.append(code)
    code_list = "\n".join(code_list)

    code = f"""
{code_list}
    """

    with open(out + ".cl", "w") as f:
        f.write(code)
