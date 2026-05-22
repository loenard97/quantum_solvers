#include "_Definitions.hpp"
#include "_Propagators.hpp"
#include "_KraussOperators.hpp"

class EBCEND {
public:
    Vector r;
    std::vector<Vector> last_rs;

    EBCEND() = default;

    std::vector<Vector> run(std::vector<Vector> Omegas, double step, size_t n_steps) {
        std::vector<Vector> mean_rs(n_steps, Vector::Zero());
        Vector cur_r;

        for (size_t i = 0; i < Omegas.size(); ++i) {
            cur_r = r;

            for (size_t t = 0; t < n_steps; ++t) {
                cur_r = bloch_rotation(cur_r, Omegas[i], step);
                mean_rs[t] += (cur_r - mean_rs[t]) / double(i + 1);
            }
        }

        r = mean_rs[n_steps];
        last_rs = mean_rs;

        return mean_rs;
    }
};