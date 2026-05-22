#include "_Definitions.hpp"
#include "_Propagators.hpp"
#include "_KraussOperators.hpp"


class EDCCKD {
public:
    double gamma_cooling = 0.0;
    double gamma_heating = 0.0;
    double gamma_dephasing = 0.0;
    Matrix rho;
    std::vector<Matrix> last_rhos;

    EDCCKD() = default;

    py::tuple as_bloch_vector()
    {
        return _density_to_bloch(last_rhos);
    }

    std::vector<Matrix> run(
        std::vector<Vector> Omegas, 
        double step, 
        size_t n_steps
    ) {
        Matrix U;
        Matrix cur_rho;
        std::vector<Matrix> mean_rhos(n_steps + 1, Matrix::Zero());
        mean_rhos[0] = rho;
        std::tuple<Matrix, Matrix> Kc = krauss_cooling(gamma_cooling, 0.5 * step);
        std::tuple<Matrix, Matrix> Kh = krauss_heating(gamma_heating, 0.5 * step);
        std::tuple<Matrix, Matrix> Kd = krauss_dephasing(gamma_dephasing, 0.5 * step);

        for (size_t i = 0; i < Omegas.size(); ++i) {
            cur_rho = rho;
            U = cayley_propagator(Omegas[i], step);

            for (size_t t = 1; t < n_steps + 1; ++t) {
                apply_krauss_operators(cur_rho, Kc);
                apply_krauss_operators(cur_rho, Kh);
                apply_krauss_operators(cur_rho, Kd);

                apply_propagator(cur_rho, U);

                apply_krauss_operators(cur_rho, Kc);
                apply_krauss_operators(cur_rho, Kh);
                apply_krauss_operators(cur_rho, Kd);

                mean_rhos[t] += (cur_rho - mean_rhos[t]) / double(i + 1);
            }
        }

        rho = mean_rhos[n_steps + 1];
        last_rhos = mean_rhos;

        return mean_rhos;
    }
};