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
t.title = 'Paramètres';

const s1 = slider({values: [0.0], min: -1, max: 1, pips: true, pipstep: 50});
const s2 = slider({values: [0.0], min: -1, max: 1, pips: true, pipstep: 50});
const s3 = slider({values: [0.0], min: -1, max: 1, pips: true, pipstep: 50});
s1.title = '';
s2.title = '';
s3.title = '';

const tog = toggle({ text: '' });
tog.title = 'Filtre passe-bas';

const togStream = toggle({ text: '' });
togStream.title = 'Stream';

const slid = slider({values: [120.0], min: 5.0, max: 500.0, step: 1.0, pips: true, pipstep: 100});
slid.title = 'Fréquence de coupure';

const sel_spa = select({ options: ['25.', '33.', '40.', '50.', '66.', '75.', '100.', '500.', '1000.', '10000.'], value: '50.' });
sel_spa.title = 'Rayon';

const sel_sam = select({ options: ['4', '5'], value: '4' });
sel_sam.title = 'Discrétisation';

const sel_mod = select({ options: ['Individuel', 'Global'], value: 'Global' });
sel_mod.title = 'Mode';

const b = button({ text: 'Lancer l\'échantillonnage' });
b.title = '';

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
  sel_spa.$value.subscribe((x) => {
    ws.send(JSON.stringify({ action: 'space_factor', value: x}));
  });
  sel_sam.$value.subscribe((x) => {
    ws.send(JSON.stringify({ action: 'sampling_factor', value: x}));
  });
  sel_mod.$value.subscribe((x) => {
    ws.send(JSON.stringify({ action: 'sampling_mode', value: x}));
  });

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
};

togStream.$checked.subscribe((x) => {
  console.log(x, testStream);
});


b.$click.subscribe(() => {
  // ws.send(JSON.stringify({ action: 'space_factor', space_factor: s3.$values.value}));
  ws.send(JSON.stringify({ action: 'sample'}));
  // ws.send(JSON.stringify({ action: 'sample_pca_individual'}));
});

const imgUpload = imageUpload();
imgUpload.title = '';
const instanceViewer = imageDisplay(imgUpload.$images);
instanceViewer.title = 'Série d\'activités ventilatoires artificielles (n=64)';


const dash = dashboard({
  title: 'GANspire',
  author: 'Marcelle Pirates Crew',
});


if (type_interface === 'standard') {
  dash.page(' ').sidebar(t, s1, s2, s3).use(gc);
} else if (type_interface === 'sampling') {
  dash.page(' ').sidebar(t, s1, s2, s3, sel_spa, sel_sam, b).use(gc);
} else if (type_interface === 'multipage') {
  dash.page('Interface').sidebar(t, s1, s2, s3, tog, slid, togStream).use(gc);
  // dash.page('Échantillonnage').sidebar(sel_spa, sel_sam, sel_mod, b, imgUpload).use(instanceViewer);
}


dash.show();
