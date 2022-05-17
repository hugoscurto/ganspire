# GANspire Python module

## API

### Constructor 

`GANspire(...)`

Parameters
- `wavegan_path`: path to folder where the pretrained model is placed. 
- `index_training`: index of training related to the pretrained model (to be removed). 

Exemple:
```
gs = GANspire(./train, 7) 
```

### Generation

`generate_from_coordinates(...)`

Parameters:
- `coordinates`: a vector of dimension 3 providing a point in the latent space used to generate a breathing waveform

Output:
- `waveform`: a 16-second breathing waveform sampled at 1000Hz

Exemple:
```
 wav = gs.generate_from_coordinates([0.0, 0.0, 0.0]) 
```



