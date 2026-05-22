#include "_Definitions.hpp"
#include "_Propagators.hpp"
#include "_KraussOperators.hpp"

class SBCEND {
public:
    Vector r;
    std::vector<Vector> last_rs;

    SBCEND() = default;

    std::vector<Vector> run(Vector Omega, double step, size_t n_steps) {
        std::vector<Vector> rs;

        for (std::size_t i = 0; i < n_steps; ++i) {
            r = bloch_rotation(r, Omega, step);
            rs.push_back(r);
        }

        last_rs = rs;

        return rs;
    }
};