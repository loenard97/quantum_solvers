#include "_Definitions.hpp"
#include "_Propagators.hpp"
#include "_KraussOperators.hpp"


class EDCEKD {
public:
    double gamma_cooling = 0.0;
    double gamma_heating = 0.0;
    double gamma_dephasing = 0.0;
    Matrix rho;

    EDCEKD() = default;

    template<typename Iterator>
    inline std::vector<Matrix> run_chunk(
        const Iterator OmegasBegin,
        const Iterator OmegasEnd, 
        const Matrix rho_0,
        const double step, 
        const size_t n_steps
    ) {
        Matrix U;
        Matrix cur_rho;
        std::vector<Matrix> mean_rhos(n_steps, Matrix::Zero());
        std::tuple<Matrix, Matrix> Kc = krauss_cooling(gamma_cooling, 0.5 * step);
        std::tuple<Matrix, Matrix> Kh = krauss_heating(gamma_heating, 0.5 * step);
        std::tuple<Matrix, Matrix> Kd = krauss_dephasing(gamma_dephasing, 0.5 * step);

        size_t i = 0;
        for (auto omega = OmegasBegin; omega != OmegasEnd; ++omega) {
            cur_rho = rho_0;
            U = su2_propagator(*omega, step);

            for (size_t t = 0; t < n_steps; ++t) {
                apply_krauss_operators(cur_rho, Kc);
                apply_krauss_operators(cur_rho, Kh);
                apply_krauss_operators(cur_rho, Kd);
                apply_propagator(cur_rho, U);
                apply_krauss_operators(cur_rho, Kc);
                apply_krauss_operators(cur_rho, Kh);
                apply_krauss_operators(cur_rho, Kd);
                mean_rhos[t] += (cur_rho - mean_rhos[t]) / double(i + 1);
            }

            ++i;
        }

        return mean_rhos;
    }

    Result run(
        const std::vector<Vector>& Omegas, 
        const double step, 
        const size_t n_steps
    ) {
        Result result;
        result.steps.push_back(step);
        result.n_steps.push_back(n_steps);

        size_t n_threads = std::thread::hardware_concurrency();
        size_t chunk_size = Omegas.size() / n_threads;
        std::vector<Matrix> temp;
        std::vector<std::vector<Matrix>> res(n_threads);
        std::vector<std::thread> threads;
        auto begin = Omegas.cbegin();
        auto end = Omegas.cend();

        if (Omegas.size() < 1000) {
            temp = run_chunk(begin, end, rho, step, n_steps);
            result.rhos.insert(result.rhos.end(), temp.begin(), temp.end());

            return result;
        }

        for (size_t i = 0; i < n_threads; ++i) {
            size_t start = i * chunk_size;
            size_t stop = (i == n_threads - 1) ? Omegas.size() : (i + 1) * chunk_size;
            begin = Omegas.cbegin() + start;
            end = Omegas.cbegin() + stop;

            threads.emplace_back([&, i, begin, end]() {
                res[i] = run_chunk(begin, end, rho, step, n_steps);
            });
        }

        for (auto& t : threads) {
            t.join();
        }

        Matrix mean = Matrix::Zero();
        temp.clear();
        for (size_t j = 0; j < res[0].size(); ++j) {
            mean = Matrix::Zero();
            for (size_t i = 0; i < res.size(); ++i) {
                mean += res[i][j];
            }
            temp.push_back(mean / n_threads);
        }
        result.rhos.insert(result.rhos.end(), temp.begin(), temp.end());

        return result;
    }
};