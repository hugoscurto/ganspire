import asyncio
from glob import glob
import json
import websockets
import math

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import itertools

from sklearn.decomposition import PCA
from scipy.io import wavfile
from scipy.signal import butter, lfilter

from pythonosc import udp_client

import os
import glob

# from IPython.display import display, Audio

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def load_graph(index_training):
    # Load the graph
    tf.reset_default_graph()
    saver = tf.train.import_meta_graph(
        wavegan_path + '/train_' + str(index_training) + '/infer/infer.meta')
    graph = tf.get_default_graph()
    sess = tf.InteractiveSession()
    saver.restore(sess, wavegan_path + '/train_' +
                  str(index_training) + '/' + name_model)

    return sess, graph


def sample_random_latent_vectors(num_samples, coeff):
    # Create num_samples random latent vectors z
    _z = (np.random.rand(num_samples, wavegan_latent_dim) * 2.) - 1
    _z = _z * coeff

    return _z


def produce_feature_vectors(_z, index_layer):
    # Synthesize G/upconv_/strided_slice(z)
    layer_name = 'G/upconv_' + str(index_layer) + '/strided_slice:0'

    z = graph.get_tensor_by_name('z:0')
    upconv = graph.get_tensor_by_name(layer_name)
    _upconv = sess.run(upconv, {z: _z})

    return _upconv


# def plot_feature_vectors(upconv, mode, size):
#     # Plot audio in notebook
#     if size == 'all':
#         fig = plt.figure(figsize=(20, 10))
#         for index in range(48):
#             ax = fig.add_subplot(6, 8, index + 1)
#             if mode == 'sample':
#                 plt.plot(upconv[index, :, 0])
#             elif mode == 'tensor':
#                 plt.plot(upconv[0, :, index])
#             ax.xaxis.set_visible(False)
#             ax.yaxis.set_visible(False)
#             ax.set_ylim(-1.5, 1.5)
#             plt.subplots_adjust(wspace=0.0)
#             plt.title(index)
#     elif size == 'zoom':
#         for index in range(48):
#             fig = plt.figure(figsize=(20, 10))
#             if mode == 'sample':
#                 plt.plot(upconv[index, :, 0])
#             elif mode == 'tensor':
#                 plt.plot(upconv[0, :, index])
#             plt.grid()

#     plt.show()


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


def compute_space_coordinates(_z, U, n_space_samples, space_factor):
    n_components = U.shape[0]
    # wavegan_latent_dim = _z.shape[1]
    z_ref = np.zeros(wavegan_latent_dim)

    z_prime = []
    space_coordinates = []
    index_prime = 0

    if n_components == 2:
        space_list = [space_factor * (np.array(item) * 2. - (n_space_samples - 1)) / (n_space_samples - 1)
                      for item in itertools.product(range(n_space_samples), repeat=n_components)]
    elif n_components == 3:
        space_list = [space_factor * (np.array(item[::-1]) * 2. - (n_space_samples - 1)) / (
            n_space_samples - 1) for item in itertools.product(range(n_space_samples), repeat=n_components)]

    for space_coordinates in space_list:
        temp = z_ref + np.dot(U.T, space_coordinates)
        z_prime.append(temp)

    return z_prime, space_list


def compute_z_prime(U, space_coordinates, space_factor):
    z_ref = np.zeros(wavegan_latent_dim)
    z_prime = [z_ref + np.dot(U.T, space_coordinates * space_factor)]

    return z_prime


def compute_space_samples(z_prime, graph):
    z = graph.get_tensor_by_name('z:0')
    G_z = graph.get_tensor_by_name('G_z:0')
    _G_z = sess.run(G_z, {z: z_prime})

    return _G_z


# def plot_space_samples(_G_z, n_space_samples, n_components, quality, index_layer, space_factor, is_save):
#     if quality == 'hd':
#         factor = 100
#     elif quality == 'sd':
#         factor = 10

#     fig = plt.figure(figsize=(2 * factor, 1 * factor))

#     def save(n_space_samples, n_components, index_sample, plt):
#         temp = (((index_sample - 1) // (n_space_samples ** 2)) *
#                 2 - (n_space_samples - 1)) / (n_space_samples - 1)
#         plt.savefig('./space/' + str(index_training) + '/layer=' + str(index_layer) +
#                     '-n_components=' + str(n_components) + '-depth=' + str(temp) + '.png')

#     for index_sample in range(n_space_samples ** n_components):

#         if index_sample % (n_space_samples ** 2) == 0:
#             if is_save:
#                 if index_sample != 0:
#                     save(n_space_samples, n_components, index_sample, plt)

#             fig = plt.figure(figsize=(2 * factor, 1 * factor))

#         plt.plot(_G_z[index_sample, :, 0])
#         plt.subplots_adjust(wspace=0.0)
#         plt.subplots_adjust(hspace=0.0)

#     if is_save:
#         save(n_space_samples, n_components, index_sample, plt)

#     plt.show()


# def generate_random_waveform():
#     phase = random.random()
#     freq = 0.2
#     sampling_freq = 1000.
#     slice_len = 16384

#     waveform = list(random.random() * np.array([math.sin(
#         2 * math.pi * (freq * i / sampling_freq + phase)) for i in range(slice_len)]))
#     waveform.append(-1.)
#     waveform.append(1.)
#     waveform.insert(0, -1.)
#     waveform.insert(0, 1.)

#     return waveform


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


def load_pca():
    with open('pca_data.txt') as json_file:
        pca_data = json.load(json_file)
        for p in pca_data['pca_data']:
            U = p['U']
            # print('explained_variance_ratio_: ', p['explained_variance_ratio_'])

    return U


def save_png(_G_z, coeff, sampling_factor, space_factor, index_sample, filter_toggle, filter_cutoff):
    fig = plt.figure(figsize=(20, 10))
    # plt.plot(_G_z[0, :, 0])
    # plt.xlim(0, len(_G_z[0, :, 0]))
    plt.plot(_G_z[0, 384:len(_G_z[0, :, 0]), 0])
    plt.xlim(0, len(_G_z[0, :, 0]) - 384)
    plt.ylim(-coeff, coeff)
    plt.xlabel('Temps (ms)')
    #plt.ylabel('voltage (mV)')
    plt.grid()
    if filter_toggle:
        str_filter = '.f' + str(int(filter_cutoff))
    else:
        str_filter = ''
    plt.savefig('./samplings/' + str(sampling_factor) +
                '/r' + str(int(space_factor)) + str_filter + '.' + str(index_sample) + '.png')

    # plt.show()


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


async def main(websocket, path):
    # Initialise message
    old_message = {}

    num_components = 3
    filter_toggle = False
    filter_cutoff = 120.

    # Define sample rate (Hz) and order of filter
    order = 5
    fs = 1000.0

    shouldStop = False
    async for s in websocket:
        message = json.loads(s)

        if message != old_message:
            print(message)

            if message["action"] == "data":
                space_coordinates = np.array(
                    [message["slider1"][0], message["slider2"][0], message["slider3"][0]])

                # Edit new image z by varying PCA coordinates x: z' = z + Ux
                z_prime = compute_z_prime(U, space_coordinates, space_factor)

                # Compute space samples
                _G_z = compute_space_samples(z_prime, graph)
                wav = _G_z[0, :, 0]

                # Filter waveform
                if filter_toggle:
                    wav = butter_lowpass_filter(wav, filter_cutoff, fs, order)

                # Convert waveform into list
                waveform = wav.tolist()

                # Send to TouchDesigner
                for index in range(10):
                    start_index = index * 1638
                    stop_index = (index + 1) * 1638
                    client.send_message(
                        "/" + str(index), waveform[start_index:stop_index])

                # Scale for Marcelle
                waveform.append(-1.)
                waveform.append(1.)
                waveform.insert(0, -1.)
                waveform.insert(0, 1.)
                # print('waveform', waveform)

                # Send to Marcelle
                await websocket.send(json.dumps(
                    {'waveform': waveform}))

            elif message["action"] == "stop":
                shouldStop = True

            elif message["action"] == "space_factor":
                space_factor = float(message["value"])

            elif message["action"] == "sampling_factor":
                sampling_factor = int(message["value"])

            elif message["action"] == "sampling_mode":
                sampling_mode = message["value"]

            elif message["action"] == "sample":
                list_sample = [(2. * np.array(list(tup)) - (sampling_factor - 1)) / (sampling_factor - 1) for tup in itertools.product(
                    range(sampling_factor), repeat=num_components)]
                print(len(list_sample))
                print('list_sample', list_sample)

                # Check sampling mode
                if sampling_mode == 'Individuel':

                    # Loop over sampled coordinates
                    index_sample = 0
                    for space_coordinates in list_sample:

                        # Edit new image z by varying PCA coordinates x: z' = z + Ux
                        z_prime = compute_z_prime(
                            U, space_coordinates, space_factor)

                        # Compute space samples
                        _G_z = compute_space_samples(z_prime, graph)

                        # Filter waveform
                        if filter_toggle:
                            _G_z[0, :, 0] = butter_lowpass_filter(
                                _G_z[0, :, 0], filter_cutoff, fs, order)

                        # Save plotted waveforms
                        coeff = 1.
                        save_png(_G_z, coeff, sampling_factor,
                                 space_factor, index_sample, filter_toggle, filter_cutoff)

                        index_sample += 1

                # Check sampling mode
                elif sampling_mode == 'Global':

                    # Plot audio in notebook
                    fig = plt.figure(figsize=(20, 10))

                    # Loop over sampled coordinates
                    index_sample = 0
                    for space_coordinates in list_sample:

                        # Edit new image z by varying PCA coordinates x: z' = z + Ux
                        z_prime = compute_z_prime(
                            U, space_coordinates, space_factor)

                        # Compute space samples
                        _G_z = compute_space_samples(z_prime, graph)

                        # Filter waveform
                        if filter_toggle:
                            _G_z[0, :, 0] = butter_lowpass_filter(
                                _G_z[0, :, 0], filter_cutoff, fs, order)

                        # Increment subplot
                        coeff = 1.
                        ax = fig.add_subplot(8, 8, index_sample + 1)
                        plt.plot(_G_z[0, :, 0])
                        ax.xaxis.set_visible(False)
                        ax.yaxis.set_visible(False)
                        ax.set_ylim(-coeff, coeff)
                        plt.subplots_adjust(wspace=0.0)
                        plt.subplots_adjust(hspace=0.0)

                        index_sample += 1

                    if filter_toggle:
                        str_filter = '.f' + str(int(filter_cutoff))
                    else:
                        str_filter = ''

                    # Save plotted waveforms
                    plt.savefig('./samplings/' + str(sampling_factor) +
                                '/all/r' + str(int(space_factor)) + str_filter + '.all.png')

                print('...Sampling done!')

                # # Send to Marcelle
                # await websocket.send(json.dumps(
                #     {'waveform': waveform}))

            elif message["action"] == "filter_toggle":
                filter_toggle = message["value"]

            elif message["action"] == "filter_cutoff":
                filter_cutoff = float(message["value"][0])
                b, a = butter_lowpass(filter_cutoff, fs, order)

            else:
                await websocket.send(json.dumps(
                    {'test': 'test'}))

            old_message = message


if __name__ == "__main__":
    index_training = 7
    list_model_ckpts = [8944, 9435, 1864, 2035, 1320, 16205, 99666,
                        990, 2018, 44612, 266213, 1544, 6431, 39975, 33317, 8360]
    name_model = 'model.ckpt-' + str(list_model_ckpts[index_training - 1])

    wavegan_path = '/Users/scurto/Documents/Eu/code/wavegan-master/train'

    with open(wavegan_path + '/train_' + str(index_training) + '/args.txt', 'r') as read_obj:
        for line in read_obj:
            if 'wavegan_latent_dim' in line:
                wavegan_latent_dim = int(line[19:])
    if index_training == 3:
        wavegan_latent_dim = 10

    # Load graph
    sess, graph = load_graph(index_training)
    print("...Graph Ready!")

    # # Compute PCA
    # ganspace()
    # print("...PCA Ready!")

    # Load PCA
    U = np.array(load_pca())
    print("...PCA Ready!")

    # Start OSC client
    client = udp_client.SimpleUDPClient('localhost', 9000)

    # Start websocket server
    start_server = websockets.serve(main, "localhost", 8766)
    print("...Server Ready!")
    asyncio.get_event_loop().run_until_complete(start_server)
    asyncio.get_event_loop().run_forever()
