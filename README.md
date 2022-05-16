# ganspire-interface

This is an interface for GANspire, a generative model based on deep neural networks for human-like breathing signals. The interface is written using the [Marcelle](https://marcelle.dev) toolkit.


## Install

After cloning the repo, make sure Marcelle is installed, otherwise install it locally:

``` 
npm install @marcellejs/core
```

### Launching the python server

GANspire runs in python and communicates while the interface is a javasript app and runs in the browser. 

In order to launch the python server, just type in the terminal: 
```
python server.py
```

The python server requires a pre-trained model that has to placed in `./train/train_7`. The pretrained model can be provided on demand. 

### Launching the app

Run the app in dev mode: 
``` 
npm run dev
```
Runs the app in the development mode.
Open http://localhost:3000 to view it in the browser.

For deployment, use:
``` 
npm run build
``` 
The command builds a static copy of your site to the `build/` folder.
Your app is ready to be deployed!

