import json
import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA


def load_pca():
    with open('pca_data.txt') as json_file:
        pca_data = json.load(json_file)
        for p in pca_data['pca_data']:
            U = p['U']
    return np.array(U)


def compute_z_prime(U, space_coordinates, space_factor, wavegan_latent_dim):
    z_ref = np.zeros(wavegan_latent_dim)
    z_prime = [z_ref + np.dot(U.T, space_coordinates * space_factor)]
    return z_prime

def compute_space_samples(sess, z_prime, graph):
    z = graph.get_tensor_by_name('z:0')
    G_z = graph.get_tensor_by_name('G_z:0')
    _G_z = sess.run(G_z, {z: z_prime})
    return _G_z


# Not used for now
def sample_random_latent_vectors(num_samples, coeff, wavegan_latent_dim):
    # Create num_samples random latent vectors z
    _z = (np.random.rand(num_samples, wavegan_latent_dim) * 2.) - 1
    _z = _z * coeff

    return _z


def produce_feature_vectors(sess, graph, _z, index_layer):
    # Synthesize G/upconv_/strided_slice(z)
    layer_name = 'G/upconv_' + str(index_layer) + '/strided_slice:0'
    z = graph.get_tensor_by_name('z:0')
    upconv = graph.get_tensor_by_name(layer_name)
    _upconv = sess.run(upconv, {z: _z})
    return _upconv

def compute_pca(upconv, n_components):
    # Flatten feature vectors
    flattened_y = np.array([upconv[index_sample, :, :].flatten()
                            for index_sample in range(upconv.shape[0])])
    # Compute PCA
    pca = PCA(n_components=n_components)
    pca.fit(flattened_y)
    V = pca.components_
    mean = pca.mean_
    flattened_x = pca.fit_transform(flattened_y)
    explained_variance_ratio_ = pca.explained_variance_ratio_
    return V, mean, flattened_y, flattened_x, explained_variance_ratio_

def compute_transferred_basis(flattened_x, _z):
    U, residuals, rank, s = np.linalg.lstsq(flattened_x, _z, rcond='warn')
    return U, residuals, rank, s

def ganspace():
    num_samples = 1000
    index_layer = 0
    n_components = 3
    coeff = 1.
    # Sample N random latent vectors: z
    _z = sample_random_latent_vectors(num_samples, coeff)
    # Produce N feature tensors at the ith layer: y
    _upconv = produce_feature_vectors(_z, index_layer)
    # Compute PCA from N feature tensors: (V, µ)
    V, mean, flattened_y, flattened_x, explained_variance_ratio_ = compute_pca(
        _upconv, n_components)
    # Compute PCA coordinates for each feature tensor: x = Vt(y - µ)
    # print('flattened_x.shape', flattened_x.shape)
    # Transfer this basis to latent space by linear regression: U = argmin(sum(Uxj - zj))
    U, residuals, rank, s = compute_transferred_basis(flattened_x, _z)
    pca_data = {}
    pca_data['pca_data'] = []
    pca_data['pca_data'].append({
        'explained_variance_ratio_': explained_variance_ratio_.tolist(),
        'U': U.tolist(),
        'residuals': residuals.tolist(),
        'rank': rank.tolist(),
        's': s.tolist()
    })
    with open('pca_data.txt', 'w') as outfile:
        json.dump(pca_data, outfile)


class GANspire:

    def __init__(self, wavegan_path, index_training):
        print('Initiating GANspire')
        self.U = load_pca()
        self.space_coordinates = [0.0, 0.0, 0.0]
        self.space_factor = 50.0
        self.index_training = index_training
        self.wavegan_path = wavegan_path
        with open(self.wavegan_path + '/train_' + str(self.index_training) + '/args.txt', 'r') as read_obj:
            for line in read_obj:
                if 'wavegan_latent_dim' in line:
                    self.wavegan_latent_dim = int(line[19:])
            if self.index_training == 3:
                self.wavegan_latent_dim = 10

    def load_model(self, model_name):
        # Load the graph
        tf.reset_default_graph()
        saver = tf.train.import_meta_graph(
            self.wavegan_path + '/train_' + str(self.index_training) + '/infer/infer.meta')
        self.graph = tf.get_default_graph()
        self.sess = tf.InteractiveSession()
        saver.restore(self.sess, self.wavegan_path + '/train_' +
                    str(self.index_training) + '/' + model_name)
        return self.sess, self.graph
    
    def generate_from_coordinates(self, coordinates):
        # Edit new image z by varying PCA coordinates x: z' = z + Ux
        z_prime = compute_z_prime(self.U, coordinates, self.space_factor, self.wavegan_latent_dim)
        _G_z = compute_space_samples(self.sess, z_prime, self.graph)
        self.wav = _G_z[0, :, 0]
        return self.wav

    def get_U(self):
        return self.U
