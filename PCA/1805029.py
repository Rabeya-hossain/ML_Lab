import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal
import pandas as pd

def initialize_parameters(data, k):
    n, m = data.shape
    # np.random.seed(42)
    means = data[np.random.choice(n, k, replace=False)]
    covariances = [np.eye(m) for _ in range(k)]
    weights = np.ones(k) / k
    return means, covariances, weights

def gaussian_pdf(x, mean, covariance):
    m = len(mean)
    det_cov = np.linalg.det(covariance)
    inv_cov = np.linalg.inv(covariance)
    exponent = -0.5 * np.dot(np.dot((x - mean).T, inv_cov), (x - mean))
    coefficient = 1 / ((2 * np.pi) ** (m / 2) * det_cov ** 0.5)
    return coefficient * np.exp(exponent)

def expectation(data, means, covariances, weights):
    k = len(means)
    n = len(data)
    data_probabilities = np.zeros((n, k))

    for i in range(k):
        data_probabilities[:, i] = weights[i] * np.apply_along_axis(gaussian_pdf, 1, data, means[i], covariances[i])

    data_probabilities /= data_probabilities.sum(axis=1, keepdims=True)
    return data_probabilities

def maximization(data, data_probabilities):
    n, m = data.shape
    k = data_probabilities.shape[1]
    
    means = np.dot(data_probabilities.T, data) / data_probabilities.sum(axis=0, keepdims=True).T
    covariances = [np.dot((data - means[i]).T, data_probabilities[:, i][:, np.newaxis] * (data - means[i])) / data_probabilities[:, i].sum() for i in range(k)]
    weights = data_probabilities.sum(axis=0) / n
    
    return means, covariances, weights

def log_likelihood(data, means, covariances, weights):
    k = len(means)
    n = len(data)
    log_likelihood_value = 0.0

    for i in range(n):
        likelihood_i = np.sum([weights[j] * gaussian_pdf(data[i], means[j], covariances[j]) for j in range(k)])
        log_likelihood_value += np.log(likelihood_i)

    return log_likelihood_value

def plot_ellipse(ax, mean, covariance, color, label):
    v, w = np.linalg.eigh(covariance)
    v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
    u = w[0] / np.linalg.norm(w[0])

    ellipse = Ellipse(xy=mean, width=v[0], height=v[1], angle=np.degrees(np.arctan(u[1] / u[0])),
                      color=color, alpha=0.2, label=label)
    ax.add_patch(ellipse)


def plot_gaussian_contour(ax, mean, covariance, color, label):
    n_std = 3  # Number of standard deviations for contour
    v, w = np.linalg.eigh(covariance)
    v = 2.0 * n_std * np.sqrt(v)
    u = w[0] / np.linalg.norm(w[0])

    # Create a grid of points
    angle = np.arctan(u[1] / u[0])
    theta = np.degrees(angle)
    # ellipse = Ellipse(mean, v[0], v[1], theta, color=color, alpha=0.2, label=label)
    # ax.add_patch(ellipse)

    # Plot contour
    x, y = np.mgrid[mean[0] - n_std * np.sqrt(covariance[0, 0]):mean[0] + n_std * np.sqrt(covariance[0, 0]):100j,
                    mean[1] - n_std * np.sqrt(covariance[1, 1]):mean[1] + n_std * np.sqrt(covariance[1, 1]):100j]
    pos = np.dstack((x, y))
    rv = multivariate_normal(mean, covariance)
    contour = ax.contour(x, y, rv.pdf(pos), colors=color, linestyles='dashed', linewidths=1)

    return contour.collections[0]


def gmm(data, k, n_iterations=5, plot_update_interval=1):
    best_means, best_covariances, best_weights, best_responsibilities,best_log_likelihood_values = None,None, None, None, None
    best_log_likelihood = float('-inf')

    plt.ion()
    fig, (ax2) = plt.subplots(1, 1, figsize=(12, 5))
    # ax1.scatter(data[:, 0], data[:, 1], alpha=0.6)
    # ax1.set_title('Data Points')

    for _ in range(2): 
        means, covariances, weights = initialize_parameters(data, k)
        log_likelihood_values = []

        for iteration in range(n_iterations):
            data_probabilities = expectation(data, means, covariances, weights)

            means, covariances, weights = maximization(data, data_probabilities)

            log_likelihood_value = log_likelihood(data, means, covariances, weights)
            log_likelihood_values.append(log_likelihood_value)

            if iteration % plot_update_interval == 0 or iteration == n_iterations - 1:
                ax2.clear()
                ax2.scatter(data[:, 0], data[:, 1], c=data_probabilities.argmax(axis=1), cmap='viridis', alpha=0.6)
                ax2.scatter(means[:, 0], means[:, 1], c='red', marker='x', s=100)
                for j in range(k):
                    plot_gaussian_contour(ax2, means[j], covariances[j], color='green', label=f'Contour {j + 1}')
                ax2.set_title(f'GMM Clustering - Iteration {iteration + 1} for k = ' + str(k) + ' Components')
                ax2.set_xlabel('Principal Component 1')
                ax2.set_ylabel('Principal Component 2')
                # ax2.legend(loc='upper right')
                plt.pause(0.1)

        if log_likelihood_value > best_log_likelihood:
            best_log_likelihood = log_likelihood_value
            best_log_likelihood_values = log_likelihood_values
            best_means, best_covariances, best_weights, best_responsibilities = means, covariances, weights, data_probabilities

    ax2.clear()
    ax2.scatter(data[:, 0], data[:, 1], c=best_responsibilities.argmax(axis=1), cmap='viridis', alpha=0.6)
    ax2.scatter(best_means[:, 0], best_means[:, 1], c='red', marker='x', s=100)
    for j in range(k):
        plot_gaussian_contour(ax2, best_means[j], best_covariances[j], color='green', label=f'Contour {j + 1}')
    ax2.set_title('Best GMM Clustering for k = ' + str(k) + ' Components')
    ax2.set_xlabel('Principal Component 1')
    ax2.set_ylabel('Principal Component 2')
    # ax2.legend(loc='upper right')
    plt.ioff()
    plt.show()

    return best_means, best_covariances, best_weights, best_responsibilities, best_log_likelihood_values



# file_path = input('Enter the path to the CSV file: ')
# file_path = file_path + ".txt"
file_path = '2D_data_points_1.txt'
print(file_path)
data = pd.read_csv(file_path, delimiter=',')

data_2d = data.to_numpy()
n , m = data.shape
if m > 2: #do PCA
    mean = np.mean(data, axis=0)
    std_dev = np.std(data, axis=0)
    standardized_data = (data - mean) / std_dev
    U, S, Vt = np.linalg.svd(standardized_data, full_matrices=False)
    data_2d = np.dot(standardized_data, Vt[:2, :].T)

# data_standardized = (data - data.mean()) / data.std()
# cov_matrix = np.cov(data_standardized, rowvar=False)

# eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

# sorted_indices = np.argsort(eigenvalues)[::-1]
# top_eigenvectors = eigenvectors[:, sorted_indices[:2]]
# projection_matrix = top_eigenvectors

# data_2d = np.dot(data_standardized, projection_matrix)

output_file_path = 'output_2d_data.csv'
pd.DataFrame(data_2d, columns=['Principal Component 1', 'Principal Component 2']).to_csv(output_file_path, index=False)


plt.scatter(data_2d[:, 0], data_2d[:, 1])
plt.title('Scatter Plot along Principal Axes')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
output_image_path = 'scatter_plot.png'
plt.savefig(output_image_path)
# plt.show()

# k = int(input('Enter the number of components: '))

k_values = range(3, 9)
log_likelihoods_per_k = []
for k in k_values:
    best_means, best_covariances, best_weights, best_responsibilities, log_likelihood_values = gmm(data_2d, k)
    log_likelihoods_per_k.append(log_likelihood_values[-1])
# best_means, best_covariances, best_weights, best_responsibilities, log_likelihood_values = gmm(data_2d, k)
    # plt.plot(range(1, len(log_likelihood_values) + 1), log_likelihood_values, marker='o')
    # plt.title('Log-Likelihood over Iterations (Best Run)')
    # plt.xlabel('Iteration')
    # plt.ylabel('Log-Likelihood')

    # output_image_path = 'log.png'
    # plt.savefig(output_image_path)

plt.plot(k_values, log_likelihoods_per_k, marker='o')
plt.title('Convergence Log-Likelihood vs. Number of Components (K)')
plt.xlabel('Number of Components (K)')
plt.ylabel('Convergence Log-Likelihood')
plt.grid(True)
output_image_path = 'log.png'
plt.savefig(output_image_path)

# plt.show()