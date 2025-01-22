#pragma once

#include <Eigen/Dense>
#include <unsupported/Eigen/Splines>
#include <vector>
#include <functional>
#include <random>
#include <algorithm>

struct DialConfig
{
    size_t seed = 0;
    size_t n_steps = 100;
    size_t Nsample = 2048;               // Number of samples
    size_t Hsample = 25;                 // Horizon of samples
    size_t Hnode = 4;                    // Node number for control
    size_t Ndiffuse = 2;                 // Number of diffusion steps
    size_t Ndiffuse_init = 10;           // Number of diffusion steps for initial diffusion
    double temp_sample = 0.05;           // Temperature for sampling
    double horizon_diffuse_factor = 0.5; // Facror to scale the sigma of horizon diffuse
    double traj_diffuse_factor = 0.5;    // Facror to scale the sigma of trajectory diffuse

    double dt = 0.02;
    double timesetep = 0.02;
    double action_scale = 1.0;
};

struct Info
{
    Eigen::VectorXd rews;
    Eigen::MatrixXd qbar;
    Eigen::MatrixXd qdbar;
    std::vector<Eigen::MatrixXd> xbar;
    Eigen::VectorXd new_noise_scale;
};

// np.einsum("n,nij->ij", weights, Y0s)
Eigen::MatrixXd computeYbar(const Eigen::VectorXd &weights, const std::vector<Eigen::MatrixXd> &Y0s)
{
    // Ensure that the size of weights matches the number of matrices in Y0s
    assert(weights.size() == Y0s.size() && "Size of weights must match number of Y0s matrices.");

    // Determine the dimensions I and J from the first matrix in Y0s
    if (Y0s.empty())
    {
        throw std::invalid_argument("Y0s vector is empty.");
    }
    const Eigen::Index I = Y0s[0].rows();
    const Eigen::Index J = Y0s[0].cols();

    // Initialize Ybar with zeros
    Eigen::MatrixXd Ybar = Eigen::MatrixXd::Zero(I, J);

    // Iterate over each n and accumulate the weighted matrices
    for (Eigen::Index n = 0; n < weights.size(); ++n)
    {
        Ybar += weights[n] * Y0s[n];
    }

    return Ybar;
}

/**
 * @brief Computes the weighted sum of a collection of matrices using provided weights.
 *
 * This function mirrors the functionality of the Python `softmax_update` function
 * implemented with JAX. It computes a weighted sum across the first dimension (`n`)
 * of the `Y0s` tensor using the `weights` vector.
 *
 * @param weights A 1D Eigen vector of length `n` containing the weights.
 * @param Y0s A vector of `n` Eigen matrices, each of shape `(i, j)`.
 * @param sigma An Eigen matrix representing `sigma` (unchanged).
 * @param mu_0t An Eigen matrix representing `mu_0t` (unused in computation).
 *
 * @return A pair containing:
 *         - `mu_0tm1`: The resulting Eigen matrix of shape `(i, j)` after the weighted sum.
 *         - `sigma`: The unchanged Eigen matrix `sigma`.
 */
std::pair<Eigen::MatrixXd, Eigen::MatrixXd> softmax_update(
    const Eigen::VectorXd &weights,
    const std::vector<Eigen::MatrixXd> &Y0s,
    const Eigen::VectorXd &sigma,
    const Eigen::MatrixXd &mu_0t)
{
    Eigen::MatrixXd mu_0tm1 = computeYbar(weights, Y0s);

    // Return the resulting mu_0tm1 and the unchanged sigma
    return std::make_pair(mu_0tm1, sigma);
}

// Helper function to map 4D indices to 1D index (row-major order)
inline size_t getIndex4D(size_t n, size_t i, size_t j, size_t k,
                         size_t N, size_t I, size_t J, size_t K)
{
    return n * (I * J * K) + i * (J * K) + j * K + k;
}

// Helper function to map 3D indices to 1D index (row-major order)
inline size_t getIndex3D(size_t i, size_t j, size_t k,
                         size_t I, size_t J, size_t K)
{
    return i * (J * K) + j * K + k;
}

// Function to compute xbar = einsum("n,nijk->ijk", weights, xss)
std::vector<double> compute_xbar(const Eigen::VectorXd &weights,
                                 const std::vector<double> &xss,
                                 size_t N, size_t I, size_t J, size_t K)
{
    // Initialize xbar as a 1D vector representing a 3D array
    std::vector<double> xbar(I * J * K, 0.0);

    // Iterate over n
    for (size_t n = 0; n < N; ++n)
    {
        double w = weights[n];
        // Iterate over i, j, k
        for (size_t i = 0; i < I; ++i)
        {
            for (size_t j = 0; j < J; ++j)
            {
                for (size_t k = 0; k < K; ++k)
                {
                    size_t xss_idx = getIndex4D(n, i, j, k, N, I, J, K);
                    size_t xbar_idx = getIndex3D(i, j, k, I, J, K);
                    xbar[xbar_idx] += w * xss[xss_idx];
                }
            }
        }
    }

    return xbar;
}

class MBDPI
{
    MBDPI(const DialConfig &config) : config_(config)
    {
        // Initialize sigmas
        double sigma0 = 1e-2;
        double sigma1 = 1e0;

        double A = sigma0;
        double B = std::log(sigma1 / sigma0) / config_.Ndiffuse;

        sigmas = Eigen::VectorXd(config_.Ndiffuse);
        for (int i = 0; i < config_.Ndiffuse; ++i)
        {
            sigmas[i] = A * std::exp(B * i);
        }

        // Initialize sigma_control
        sigma_control = Eigen::VectorXd(config_.Hnode + 1);
        for (int i = 0; i <= config_.Hnode; ++i)
        {
            sigma_control[i] = std::pow(config_.horizon_diffuse_factor, config_.Hnode - i + 1);
        }

        ctrl_dt = 0.02;
        // Initialize step_us and step_nodes
        step_us = Eigen::VectorXd::LinSpaced(config.Hsample + 1, 0, ctrl_dt * config.Hsample);
        step_nodes = Eigen::VectorXd::LinSpaced(config.Hnode + 1, 0, ctrl_dt * config.Hsample);
        node_dt = ctrl_dt * config.Hsample / config.Hnode;
    }

    // Function to perform spline interpolation from nodes to u
    Eigen::VectorXd node2u(const Eigen::VectorXd &nodes)
    {
        // Ensure that step_nodes and nodes have the same size
        assert(step_nodes.size() == nodes.size());

        // Create a spline interpolator (quadratic spline, k=2)
        Eigen::Spline<double, 1> spline = Eigen::SplineFitting<Eigen::Spline<double, 1>>::Interpolate(
            nodes.transpose(), 2, step_nodes.transpose());

        // Evaluate the spline at step_us
        Eigen::VectorXd us(step_us.size());
        for (int i = 0; i < step_us.size(); ++i)
        {
            us[i] = spline(step_us[i])(0);
        }

        return us;
    }

    // Function to perform spline interpolation from u to nodes
    Eigen::VectorXd u2node(const Eigen::VectorXd &us)
    {
        // Ensure that step_us and us have the same size
        assert(step_us.size() == us.size());

        // Create a spline interpolator (quadratic spline, k=2)
        Eigen::Spline<double, 1> spline = Eigen::SplineFitting<Eigen::Spline<double, 1>>::Interpolate(
            us.transpose(), 2, step_us.transpose());

        // Evaluate the spline at step_nodes
        Eigen::VectorXd nodes(step_nodes.size());
        for (int i = 0; i < step_nodes.size(); ++i)
        {
            nodes[i] = spline(step_nodes[i])(0);
        }

        return nodes;
    }

    // Vectorized node2u over the node axis (horizon, node)
    Eigen::MatrixXd node2u_vmap(const Eigen::MatrixXd &nodes)
    {
        int horizon = nodes.rows();
        int node = nodes.cols();
        int us_size = step_us.size();

        Eigen::MatrixXd us_matrix(horizon, us_size);

// Parallelize over the horizon axis
#pragma omp parallel for
        for (int i = 0; i < horizon; ++i)
        {
            Eigen::VectorXd current_nodes = nodes.row(i).transpose(); // (node)
            us_matrix.row(i) = node2u(current_nodes).transpose();
        }

        return us_matrix; // (horizon, us_size)
    }

    // Vectorized u2node over the node axis (horizon, node)
    Eigen::MatrixXd u2node_vmap(const Eigen::MatrixXd &us)
    {
        int horizon = us.rows();
        int us_size = us.cols();
        int node = step_nodes.size();

        Eigen::MatrixXd nodes_matrix(horizon, node);

// Parallelize over the horizon axis
#pragma omp parallel for
        for (int i = 0; i < horizon; ++i)
        {
            Eigen::VectorXd current_us = us.row(i).transpose(); // (us_size)
            nodes_matrix.row(i) = u2node(current_us).transpose();
        }

        return nodes_matrix; // (horizon, node)
    }

    // Double Vectorized node2u over batch and node axes (batch, horizon, node)
    std::vector<Eigen::MatrixXd> node2u_vvmap(const std::vector<Eigen::MatrixXd> &nodes_batch)
    {
        int batch_size = nodes_batch.size();
        std::vector<Eigen::MatrixXd> us_batch(batch_size);

// Parallelize over the batch axis
#pragma omp parallel for
        for (int b = 0; b < batch_size; ++b)
        {
            us_batch[b] = node2u_vmap(nodes_batch[b]);
        }

        return us_batch; // (batch, horizon, us_size)
    }

    // Double Vectorized u2node over batch and node axes (batch, horizon, us_size)
    std::vector<Eigen::MatrixXd> u2node_vvmap(const std::vector<Eigen::MatrixXd> &us_batch)
    {

        int batch_size = us_batch.size();
        std::vector<Eigen::MatrixXd> nodes_batch(batch_size);

// Parallelize over the batch axis
#pragma omp parallel for
        for (int b = 0; b < batch_size; ++b)
        {
            nodes_batch[b] = u2node_vmap(us_batch[b]);
        }

        return nodes_batch; // (batch, horizon, node)
    }

    Eigen::MatrixXd shift(const Eigen::MatrixXd &Y)
    {
        Eigen::MatrixXd u = node2u_vmap(Y);
    }

    // Function to clip a value to a specified range
    inline double clip(double x, double min_val, double max_val)
    {
        return std::max(min_val, std::min(x, max_val));
    }

    // Function to sample Y0s using a vector of Eigen matrices
    std::vector<Eigen::MatrixXd> sample_Y0s(
        std::mt19937 &rng,                  // Original RNG
        const Eigen::VectorXd &noise_scale, // Scaling factors (size Hnode + 1)
        const Eigen::MatrixXd &Ybar_i,      // Baseline matrix (size Hnode + 1 x nu)
        int Nsample                         // Number of samples to generate
    )
    {
        // Determine dimensions
        const int Hnode_plus1 = noise_scale.size();
        const int nu = Ybar_i.cols();

        // Validate dimensions
        if (Ybar_i.rows() != Hnode_plus1)
        {
            throw std::invalid_argument("Ybar_i rows must match noise_scale size.");
        }

        // Split the RNG: Initialize a new RNG engine with a seed derived from the original RNG
        std::seed_seq seed_seq{static_cast<unsigned int>(rng())};
        std::mt19937 Y0s_rng(seed_seq);

        // Define normal distribution (mean=0, stddev=1)
        std::normal_distribution<double> dist(0.0, 1.0);

        // Initialize a vector to hold Nsample matrices
        std::vector<Eigen::MatrixXd> Y0s;
        Y0s.reserve(Nsample + 1); // Reserve space for Nsample samples plus the baseline

        // Generate Nsample sampled matrices
        for (int i = 0; i < Nsample; ++i)
        {
            // Sample eps_Y: matrix of size (Hnode +1) x nu with standard normal entries
            Eigen::MatrixXd eps_Y = Eigen::MatrixXd::NullaryExpr(Hnode_plus1, nu, [&]()
                                                                 { return dist(Y0s_rng); });

            // Scale eps_Y by noise_scale (each row j scaled by noise_scale[j])
            Eigen::MatrixXd scaled_eps_Y = eps_Y;
            for (int j = 0; j < Hnode_plus1; ++j)
            {
                scaled_eps_Y.row(j) *= noise_scale(j);
            }

            // Shift by Ybar_i
            Eigen::MatrixXd Y0 = scaled_eps_Y + Ybar_i;

            // Fix the first row to match Ybar_i
            Y0.row(0) = Ybar_i.row(0);

            // Clip all values to [-1.0, 1.0] using Eigen's array operations
            Y0 = Y0.cwiseMin(1.0).cwiseMax(-1.0);

            // Add the sampled matrix to the vector
            Y0s.emplace_back(Y0);
        }

        // Append Ybar_i as an additional sample
        Eigen::MatrixXd Ybar_i_clipped = Ybar_i.cwiseMin(1.0).cwiseMax(-1.0);
        Y0s.emplace_back(Ybar_i_clipped);

        return Y0s;
    }

    std::tuple<Eigen::MatrixXd> reverse_once(Eigen::VectorXd state, auto rng, Eigen::MatrixXd Ybar_i, Eigen::VectorXd noise_scale)
    {
    }
    DialConfig config_;
    Eigen::VectorXd sigmas;
    Eigen::VectorXd sigma_control;
    Eigen::VectorXd step_us;
    Eigen::VectorXd step_nodes;
    double ctrl_dt;
    double node_dt;
};