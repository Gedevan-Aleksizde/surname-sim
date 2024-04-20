#include <iostream>
#include <vector>
#include <random>
#include <numeric>
#include <functional>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_errno.h>
// #include <boost/math/distributions/hypergeometric.hpp>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

std::vector<uint64_t> randomMultNomImpl(
    std::vector<uint64_t> numberPerName,
    std::mt19937_64 &rng) {
    double r = 1;
    const int numNames = numberPerName.size();
    const int population = std::accumulate(numberPerName.begin(), numberPerName.end(), 0);
    int numPairs = static_cast<int>(population / 2);
    std::vector<uint64_t> x(numNames, 0);
    std::vector<double> p(numNames, 0);
    double denom = std::accumulate(
        numberPerName.begin(),
        numberPerName.end(),
        decltype(numberPerName)::value_type(0));
    for (int i = 0; i < numNames; i++) {
        p[i] = numberPerName[i] / denom;
    }
    for (int i = 0; i < numNames - 1; i++) {
        if (r != 0) {
            std::binomial_distribution<int> binom(numPairs, p[i] / r);
            x[i] = binom(rng);
        } else {
            x[i] = 0;
        }
        r = r - p[i];
        numPairs = numPairs - x[i];
    }
    x[numNames - 1] = numPairs;
    std::transform(
        x.begin(), x.end(), x.begin(),
        std::bind(std::multiplies<uint64_t>(), std::placeholders::_1, 2));
    return x;
}

// https://stackoverflow.com/questions/53196221/producing-random-variates-distributed-hypergeometrically
uint64_t randomHyperGeometricShuffle(
    const uint64_t& K,
    const uint64_t& N,
    const uint64_t& n,
    std::mt19937_64&  rng
    ) {
    if (n > N) {
        throw std::runtime_error("n=" + std::to_string(n) +  " > N=" + std::to_string(N) +"!");
    }
    std::vector<uint64_t> vec(N);
    std::iota(vec.begin(), vec.end(), 0);
    uint64_t k = 0;
    for (uint64_t i = 0; i < n; i++) {
        std::uniform_int_distribution<uint64_t> unif(i, N-1);
        uint64_t s = unif(rng);
        std::swap(vec[i], vec[s]);
        k += vec[i] < K ? 1 : 0;
    }
    return k;
}

std::vector<uint64_t> randomMultHGImpl(
    const std::vector<uint64_t> numberPerName,
    std::mt19937_64&  rng
    ) {
    const int numNames = numberPerName.size();
    uint64_t population = std::accumulate(numberPerName.begin(), numberPerName.end(), 0);
    uint64_t numPairs = static_cast<int>(population / 2);
    std::vector<uint64_t> x(numNames, 0);
    for (int i = 0; i < numNames - 1; i++) {
        if (population > numPairs) {
            x[i] = randomHyperGeometricShuffle(
                numberPerName[i],
                population,
                numPairs,
                rng);
        } else {
            x[i] = numPairs;
        }
        population -= numberPerName[i];
        numPairs -= x[i];
    }
    x[numNames - 1] = numPairs;
    std::transform(
        x.begin(), x.end(), x.begin(),
        std::bind(std::multiplies<int>(), std::placeholders::_1, 2));
    return x;
}

std::vector<uint64_t> randomMultHGIndexImpl(
    const std::vector<uint64_t> numberPerName,
    std::mt19937_64&  rng
    ) {
    const int numNames = numberPerName.size();
    const uint64_t population = std::accumulate(numberPerName.begin(), numberPerName.end(), 0);
    const int numPairs = static_cast<int>(population / 2);
    std::vector<uint64_t> x(numNames, 0);
    std::vector<uint64_t> individuals(population, 0);
    std::vector<uint64_t> index(population, 0);
    std::iota(individuals.begin(), individuals.end(), 0);
    int offset = 0;
    for (int l = 0; l < numNames; l++) {
        std::fill_n(index.begin() + offset, numberPerName[l], l);
        offset += numberPerName[l];
    }
    // a drop in the bucket...
    for (int i = 0; i < numPairs; i++) {
        std::uniform_int_distribution<uint64_t> unif(i, index.size() - 1);
        int s = unif(rng);
        x[index[s]]++;
        std::swap(index[i], index[s]);
    }

    std::transform(
        x.begin(), x.end(), x.begin(),
        std::bind(std::multiplies<int>(), std::placeholders::_1, 2));
    return x;
}


std::vector<uint64_t> randomGSLMultHGImpl(
    std::vector<uint64_t> numberPerName,
    gsl_rng *rng
    ) {
    const int numNames = numberPerName.size();
    uint64_t population = std::accumulate(numberPerName.begin(), numberPerName.end(), 0);
    uint64_t numPairs = static_cast<int>(population / 2);
    std::vector<uint64_t> x(numNames, 0);
    for (int i = 0; i < numNames - 1; i++) {
        population -= numberPerName[i];
        if (numPairs > numberPerName[i]) {
            x[i] = gsl_ran_hypergeometric(rng, numberPerName[i], population, numPairs);
        } else {
            x[i] = numPairs;
        }
        population -= x[i];
        numPairs -= x[i];
    }
    x[numNames - 1] = numPairs;
    std::transform(
        x.begin(), x.end(), x.begin(),
        std::bind(std::multiplies<int>(), std::placeholders::_1, 2));
    return x;
}

py::list randomMultNomPy(std::vector<uint64_t> numberPerName, uint64_t seed = 42) {
    std::seed_seq ss{seed};
    std::mt19937_64 rng(ss);
    return py::cast(randomMultNomImpl(numberPerName, rng));
}

int randIndexWeighted(
    const std::vector<uint64_t>& weightVector,
    std::mt19937_64& rng
    ) {
    double r = 0;
    uint64_t population = std::accumulate(weightVector.begin(), weightVector.end(), 0);
    int size = weightVector.size();
    std::uniform_real_distribution<double> sampler(0, population);
    r = sampler(rng);
    int j = 0;
    for (j = 0; j < size; j++) {
            if (r < weightVector[j]) {
                break;
            }
            r -= weightVector[j];
        }
    return j;
}

std::vector<uint64_t> randomSampleWithoutRepImpl(
    std::vector<uint64_t> numberPerName,
    std::mt19937_64 &rng
    ) {
    uint64_t numNames = numberPerName.size();
    uint64_t population = std::accumulate(numberPerName.begin(), numberPerName.end(), 0);
    int numPairs = static_cast<int>(population / 2);
    std::vector<uint64_t> x(numNames, 0);
    int j = 0;
    for (int i = 0; i < numPairs; i++) {
        j = randIndexWeighted(numberPerName, rng);
        x[j] +=1;
        numberPerName[j] -= 1;
        if (numberPerName[j] == 0) {
            numberPerName.push_back(j);
        }
    }
    return x;
}


class progressBarCUI {
 private:
    int width = 20;
    float percentageDelta;
    float percentage = 0.0;  // %
    int pbState = 0;
 public:
    progressBarCUI(int iters, int width = 20) {
        width = width;
        percentageDelta = 1 / iters;
    }
    void init() {
        percentage = 0.0;
        pbState = 0;
        drawBar();
    }
    void update() {
        percentage += percentageDelta;
        if (percentage > 1.0)
            percentage = 1.0;
        pbState = percentage * width;
        drawBar();
    }
    void finish() {
        std::cout << std::endl;
    }

 private:
    void drawBar() {
        std::cout << "[";
        for (int i = 0; i < width; i++) {
            if (i < pbState)
                std::cout << "=";
            else if (i == pbState)
                std::cout << ">";
            else
                std::cout << " ";
        }
        std::cout << "] " << int(percentage * 100.0) << " %\r";
        std::cout.flush();
    }
};

class SimpleSimulator
{
 private:
    std::seed_seq ss;
    std::mt19937_64 rng;
    const gsl_rng_type *T = gsl_rng_default;
    gsl_rng *r_gsl = gsl_rng_alloc(T);

 public:
    SimpleSimulator(unsigned int seed = 42) {
        std::seed_seq ss{seed};
        rng.seed(ss);
        gsl_rng_set(r_gsl, seed);
    }
    ~SimpleSimulator() {
        gsl_rng_free(r_gsl);
    }
    void setSeed(unsigned int seed) {
        std::seed_seq ss{seed};
        rng.seed(ss);
        gsl_rng_set(r_gsl, seed);
    }
    py::list nextRandom(std::vector<uint64_t> numberPerName) {
        return py::cast(::randomMultHGIndexImpl(numberPerName, rng));
    }

 public:
    py::array_t<int> iterateHGChunk(
        std::vector<uint64_t> numberPerName,
        int nIters,
        bool onlyReturnLast = true) {
        int vecSize = numberPerName.size();
        std::vector<uint64_t> current = numberPerName;
        std::vector<std::vector<uint64_t>> result(
            onlyReturnLast == true ? 1 : nIters,
            std::vector<uint64_t>(vecSize, 0));
        if (onlyReturnLast == true) {
            for (int i = 0; i < nIters; i++) {
                current = ::randomMultHGIndexImpl(current, rng);
            }
            result[0] = current;
        } else {
            for (int i = 0; i < nIters; i++) {
                current = ::randomMultHGIndexImpl(current, rng);
                result[i] = current;
            }
        }
        return py::cast(result);
    }
    py::array_t<int> iterateMNChunk(
        std::vector<uint64_t> numberPerName,
        int nIters,
        bool onlyReturnLast = true) {
        int vecSize = numberPerName.size();
        std::vector<uint64_t> current = numberPerName;
        std::vector<std::vector<uint64_t>> result(
            onlyReturnLast == true ? 1 : nIters,
            std::vector<uint64_t>(vecSize, 0));
        if (onlyReturnLast == true) {
            for (int i = 0; i < nIters; i++) {
                current = ::randomMultNomImpl(current, rng);
            }
            result[0] = current;
        } else {
            for (int i = 0; i < nIters; i++) {
                current = ::randomMultNomImpl(current, rng);
                result[i] = current;
            }
        }
        return py::cast(result);
    }
};

PYBIND11_MODULE(surname_sim, m) {
    m.doc() = "pybind11 class";
    m.def("randomMultNomPy", &randomMultNomPy, "", py::arg("number_per_name"), py::arg("seed") = 42);
    py::class_<SimpleSimulator>(m, "SimpleSimulator")
        .def(py::init<unsigned int>(), py::arg("seed") = 42)
        .def("nextRandom", &SimpleSimulator::nextRandom, py::arg("number_per_name"))
        .def("iterateChunkHG", &SimpleSimulator::iterateHGChunk,
            py::arg("number_per_name"), py::arg("iters"), py::arg("onlyLast") = true)
        .def("iterateChunkMN", &SimpleSimulator::iterateMNChunk,
            py::arg("number_per_name"), py::arg("iters"), py::arg("onlyLast") = true);
    #ifdef VERSION_INFO
        m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
    #else
        m.attr("__version__") = "dev";
    #endif
}
