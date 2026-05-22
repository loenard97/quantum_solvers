#include "_Definitions.hpp"
#include "_Propagators.hpp"
#include "_KraussOperators.hpp"


class EDTCND {
public:
    Matrix rho;
    std::vector<Matrix> last_rhos;

    EDTCND() = default;

    py::tuple as_bloch_vector()
    {
        return _density_to_bloch(last_rhos);
    }

    std::vector<Matrix> run(
        std::vector<std::vector<Vector>> Omegas, 
        double step, 
        size_t n_steps
    ) {
        Matrix U;
        Matrix cur_rho;
        std::vector<Matrix> mean_rhos(n_steps + 1, Matrix::Zero());
        mean_rhos[0] = rho;

        for (size_t i = 0; i < Omegas.size(); ++i) {
            cur_rho = rho;

            for (size_t t = 1; t < n_steps + 1; ++t) {
                U = cayley_propagator(Omegas[i][t - 1], step);
                apply_propagator(cur_rho, U);
                mean_rhos[t] += (cur_rho - mean_rhos[t]) / double(i + 1);
            }
        }

        rho = mean_rhos[n_steps + 1];
        last_rhos = mean_rhos;

        return mean_rhos;
    }
};