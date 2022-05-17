import asyncio
import json
import websockets
import numpy as np
import matplotlib.pyplot as plt
import itertools
from scipy.signal import butter, lfilter
import time
# from pythonosc import udp_client

from ganspire import GANspire


# from IPython.display import display, Audio

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# def compute_space_coordinates(_z, U, n_space_samples, space_factor, wavegan_latent_dim):
#     n_components = U.shape[0]
#     # wavegan_latent_dim = _z.shape[1]
#     z_ref = np.zeros(wavegan_latent_dim)

#     z_prime = []
#     space_coordinates = []
#     index_prime = 0

#     if n_components == 2:
#         space_list = [space_factor * (np.array(item) * 2. - (n_space_samples - 1)) / (n_space_samples - 1)
#                       for item in itertools.product(range(n_space_samples), repeat=n_components)]
#     elif n_components == 3:
#         space_list = [space_factor * (np.array(item[::-1]) * 2. - (n_space_samples - 1)) / (
#             n_space_samples - 1) for item in itertools.product(range(n_space_samples), repeat=n_components)]

#     for space_coordinates in space_list:
#         temp = z_ref + np.dot(U.T, space_coordinates)
#         z_prime.append(temp)

#     return z_prime, space_list




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

                # get latent coordinates from sliders on the interface
                space_coordinates = np.array(
                    [message["slider1"][0], message["slider2"][0], message["slider3"][0]])

                # generate breathing signal
                wav = gs.generate_from_coordinates(space_coordinates)

                # Filter waveform
                if filter_toggle:
                    wav = butter_lowpass_filter(wav, filter_cutoff, fs, order)

                # Convert waveform into list and Scale for Marcelle
                waveform = wav.tolist()

                # Send to Marcelle
                await websocket.send(json.dumps(
                    {'waveform': waveform}))

            elif message["action"] == "filter_toggle":
                filter_toggle = message["value"]

            elif message["action"] == "filter_cutoff":
                filter_cutoff = float(message["value"][0])
                b, a = butter_lowpass(filter_cutoff, fs, order)

            # if message["action"] == "stream":
            #     # if message["status"][0]:
            #     # Send to Marcelle
            #     print(message)
            #     time_now = 0
            #     while message["status"]:
            #         # if time.time()-time_now > 1.0:
            #         await asyncio.sleep(0.1)
            #         message_ = websocket.recv()
            #         await websocket.send(json.dumps({'data': 1.0}))


            else:
                
                await websocket.send(json.dumps(
                    {'test': 'test'}))
                        
            old_message = message


if __name__ == "__main__":

    # Training paramters
    wavegan_path = './train'
    index_training = 7
    list_model_ckpts = [8944, 9435, 1864, 2035, 1320, 16205, 99666,
                        990, 2018, 44612, 266213, 1544, 6431, 39975, 33317, 8360]

    # GANspire
    gs = GANspire(wavegan_path, index_training)
    gs.load_model(model_name='model.ckpt-' + str(list_model_ckpts[index_training - 1]))
    print("...Model Ready!")

    # Start websocket server
    start_server = websockets.serve(main, "localhost", 8766)
    print("...Server Ready!")
    asyncio.get_event_loop().run_until_complete(start_server)
    asyncio.get_event_loop().run_forever()
