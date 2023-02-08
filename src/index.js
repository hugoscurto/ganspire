import '@marcellejs/core/dist/marcelle.css';
import { 
  button, 
  select, 
  text, 
  dashboard, 
  slider, 
  genericChart, 
  createStream, 
  Stream,
  toggle, 
  imageUpload, 
  imageDisplay, 
  fileUpload,
  dataStore, 
  dataset, 
  datasetBrowser } from '@marcellejs/core';


let currentWaveform = [];
const type_interface = 'multipage'; // standard, sampling, multipage

const t = text({ text: '' });
t.title = 'paramètres latents';

const s1 = slider({values: [0.0], min: -1, max: 1, pips: true, pipstep: 50});
const s2 = slider({values: [0.0], min: -1, max: 1, pips: true, pipstep: 50});
const s3 = slider({values: [0.0], min: -1, max: 1, pips: true, pipstep: 50});
s1.title = '';
s2.title = '';
s3.title = '';

const tog = toggle({ text: '' });
tog.title = 'filtre passe-bas';

const togStream = toggle({ text: '' });
togStream.title = 'flux OSC (port : 1235)';

//const label = textInput('myLabel');
//label.title = 'Instance label';
//label.$value.subscribe(console.log);

const slid = slider({values: [120.0], min: 5.0, max: 500.0, step: 1.0, pips: true, pipstep: 100});
slid.title = 'fréquence de coupure';

const sel_spa = select({ options: ['25.', '33.', '40.', '50.', '66.', '75.', '100.', '500.', '1000.', '10000.'], value: '50.' });
sel_spa.title = 'rayon';

//const sel_sam = select({ options: ['4', '5'], value: '4' });
//sel_sam.title = 'discrétisation';

//const sel_mod = select({ options: ['individuel', 'global'], value: 'global' });
//sel_mod.title = 'mode';

//const b = button({ text: 'lancer l\'échantillonnage' });
//b.title = '';

const gc = genericChart({
  preset: 'line-fast',
  options: {
    xlabel: 'x label',
    ylabel: 'y label',
  }
});
gc.title = '';

const testStream = new Stream([], false);
gc.addSeries(testStream, '');

const ws = new WebSocket('ws://127.0.0.1:8766/');

ws.onopen = () => {

  function getCoordinates() {
    ws.send(JSON.stringify({ action: 'get_coordinates', value: Math.random()}));
    // console.log('getCoordinates')
  }
  const myTimeout = setInterval(getCoordinates, 1000);

  sel_spa.$value.subscribe((x) => {
    ws.send(JSON.stringify({ action: 'space_factor', value: x}));
  });
  //sel_sam.$value.subscribe((x) => {
  //  ws.send(JSON.stringify({ action: 'sampling_factor', value: x}));
  //});
  //sel_mod.$value.subscribe((x) => {
  //  ws.send(JSON.stringify({ action: 'sampling_mode', value: x}));
  //});

  s1.$values.subscribe(() => {
    // ws.send(JSON.stringify({ action: 'data', slider1: s1.$values.value}));
    // console.log(s1.$values.value)
    ws.send(JSON.stringify({ action: 'data', slider1: s1.$values.value, slider2: s2.$values.value, slider3: s3.$values.value}));
  });
  s2.$values.subscribe(() => {
    // ws.send(JSON.stringify({ action: 'data', slider2: s2.$values.value}));
    ws.send(JSON.stringify({ action: 'data', slider1: s1.$values.value, slider2: s2.$values.value, slider3: s3.$values.value}));
  });
  s3.$values.subscribe(() => {
    // ws.send(JSON.stringify({ action: 'data', slider3: s3.$values.value}));
    ws.send(JSON.stringify({ action: 'data', slider1: s1.$values.value, slider2: s2.$values.value, slider3: s3.$values.value}));
  });

  tog.$checked.subscribe((x) => {
    ws.send(JSON.stringify({ action: 'filter_toggle', value: x}));
    ws.send(JSON.stringify({ action: 'data', slider1: s1.$values.value, slider2: s2.$values.value, slider3: s3.$values.value}));
  });

  togStream.$checked.subscribe((x) => {
    ws.send(JSON.stringify({ action: 'data', slider1: s1.$values.value, slider2: s2.$values.value, slider3: s3.$values.value}));
    ws.send(JSON.stringify({ action: 'osc_start', value: x}));
    });

  slid.$values.subscribe(() => {
    ws.send(JSON.stringify({ action: 'filter_cutoff', value: slid.$values.value}));
    setTimeout(function () {
      ws.send(JSON.stringify({ action: 'data', slider1: s1.$values.value, slider2: s2.$values.value, slider3: s3.$values.value}));
    }, 200);
  });
};


ws.onmessage = function(event) {
  let msg = JSON.parse(event.data);
  testStream.set(msg.waveform);
  
  // console.log([msg.slider1])
  if (msg.slider1 != undefined) {
    s1.$values.set([msg.slider1])
  }
  if (msg.slider2 != undefined) {
    s2.$values.set([msg.slider2])
  }
  if (msg.slider3 != undefined) {
    s3.$values.set([msg.slider3])
  }
};

togStream.$checked.subscribe((x) => {
  console.log(x, testStream);
});


//b.$click.subscribe(() => {
//  ws.send(JSON.stringify({ action: 'space_factor', space_factor: s3.$values.value}));
//  ws.send(JSON.stringify({ action: 'sample'}));
//  ws.send(JSON.stringify({ action: 'sample_pca_individual'}));
//});

const imgUpload = imageUpload();
imgUpload.title = '';
const instanceViewer = imageDisplay(imgUpload.$images);
instanceViewer.title = 'série d\'activités ventilatoires computationnelles (n=64)';


const dash = dashboard({
  title: 'GANspire',
  author: 'Hugo Scurto, Baptiste Caramiaux',
});


if (type_interface === 'standard') {
  dash.page(' ').sidebar(t, s1, s2, s3).use(gc);
} else if (type_interface === 'sampling') {
  //dash.page(' ').sidebar(t, s1, s2, s3, sel_spa, sel_sam, b).use(gc);
  dash.page(' ').sidebar(t, s1, s2, s3, sel_spa).use(gc);
} else if (type_interface === 'multipage') {
  dash.page('interface').sidebar(t, s1, s2, s3, tog, slid, togStream).use(gc);
  //dash.page('échantillonnage').sidebar(sel_spa, sel_sam, sel_mod, b, imgUpload).use(instanceViewer);
  dash.page('échantillonnage').sidebar(sel_spa, imgUpload).use(instanceViewer);
}

dash.show();
