import * as Tone from 'tone';

// Sound drawn on a canvas. Browsers start audio only after a click, hence the start button.
// To play a song made with WanGP instead of the synth: wangp.app.import_asset(song, '<app>', 'assets/music.ogg') in a Python run, then
//   const player = new Tone.Player('assets/music.ogg').toDestination(); await Tone.loaded(); player.start();
const canvas = document.getElementById('view');
const context = canvas.getContext('2d');
const start = document.getElementById('start');
let width = 0, height = 0, playing = false, step = 0, level = -Infinity;

function resize() {
  const ratio = Math.min(devicePixelRatio, 2);
  width = innerWidth; height = innerHeight;
  canvas.width = Math.round(width * ratio); canvas.height = Math.round(height * ratio);
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
}
addEventListener('resize', resize);
resize();

const analyser = new Tone.Analyser('fft', 128);
Tone.getDestination().connect(analyser);
const synth = new Tone.PolySynth(Tone.Synth, { oscillator: { type: 'triangle8' }, envelope: { attack: 0.01, release: 0.6 } }).toDestination();
const bass = new Tone.MonoSynth({ oscillator: { type: 'square' }, filter: { Q: 2, type: 'lowpass' }, filterEnvelope: { baseFrequency: 80, octaves: 3 } }).toDestination();
const notes = ['C4', 'E4', 'G4', 'B4', 'A4', 'G4', 'E4', 'D4'];
const loop = new Tone.Loop(time => {
  synth.triggerAttackRelease(notes[step % notes.length], '8n', time, 0.6);
  if (step % 4 === 0) bass.triggerAttackRelease(step % 8 === 0 ? 'C2' : 'G1', '4n', time, 0.8);
  step += 1;
}, '8n');
Tone.getTransport().bpm.value = 112;

async function toggle() {
  if (!playing) {
    await Tone.start();
    loop.start(0);
    Tone.getTransport().start();
  } else {
    Tone.getTransport().stop();
  }
  playing = !playing;
  start.hidden = playing;
  start.textContent = 'Click to start';
}
start.addEventListener('click', toggle);
canvas.addEventListener('click', () => { if (playing) toggle(); });

function draw() {
  const values = analyser.getValue();
  context.fillStyle = 'rgba(7, 8, 12, 0.35)';
  context.fillRect(0, 0, width, height);
  const bar = width / values.length;
  let sum = 0;
  values.forEach((value, index) => {
    const strength = Math.max(0, (Number(value) + 100) / 100);  // dB to 0..1
    sum += Number.isFinite(value) ? value : -100;
    const barHeight = strength * height * 0.8;
    context.fillStyle = `hsl(${190 + index * 1.4}, 85%, ${45 + strength * 25}%)`;
    context.fillRect(index * bar + 1, height - barHeight, bar - 2, barHeight);
  });
  level = sum / values.length;
  requestAnimationFrame(draw);
}
requestAnimationFrame(draw);

window.wangpState = () => ({ playing, step, level: Number.isFinite(level) ? +level.toFixed(1) : null, context: Tone.getContext().state });
