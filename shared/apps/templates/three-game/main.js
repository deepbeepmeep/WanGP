import * as THREE from 'three';

// A small arena game: move the player to collect orbs before the timer ends. Replace the rules, keep the structure:
// input state, a fixed update(dt) with a clamped step, render, HUD, saved best score, window.wangpState().
const ARENA = 9, ROUND_S = 30, ORBS = 6;
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(innerWidth, innerHeight);
document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x070b12);
scene.fog = new THREE.Fog(0x070b12, 18, 40);
const camera = new THREE.PerspectiveCamera(55, innerWidth / innerHeight, 0.1, 100);
scene.add(new THREE.HemisphereLight(0xcfe3ff, 0x1a1f2b, 1.1));
const light = new THREE.DirectionalLight(0xffffff, 2.2);
light.position.set(6, 12, 5);
scene.add(light);

const floor = new THREE.Mesh(new THREE.PlaneGeometry(ARENA * 2, ARENA * 2), new THREE.MeshStandardMaterial({ color: 0x18223a, roughness: 0.85 }));
floor.rotation.x = -Math.PI / 2;
scene.add(floor, new THREE.GridHelper(ARENA * 2, 18, 0x2f4a7a, 0x22314d));

const player = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), new THREE.MeshStandardMaterial({ color: 0x4cc9f0 }));
player.position.y = 0.5;
scene.add(player);
const orbGeometry = new THREE.SphereGeometry(0.35, 24, 16);
const orbMaterial = new THREE.MeshStandardMaterial({ color: 0xffd166, emissive: 0x6b4b00 });
const orbs = [];

const keys = new Set();
addEventListener('keydown', event => { keys.add(event.code); if (event.code === 'KeyR') restart(); if (event.code.startsWith('Arrow') || event.code === 'Space') event.preventDefault(); });
addEventListener('keyup', event => keys.delete(event.code));
addEventListener('blur', () => keys.clear());

const hud = { score: document.getElementById('score'), best: document.getElementById('best'), message: document.getElementById('message') };
let score = 0, timeLeft = ROUND_S, over = false, best = Number(localStorage.getItem('best') || 0);

function placeOrb(orb) {
  orb.position.set((Math.random() * 2 - 1) * (ARENA - 1), 0.5, (Math.random() * 2 - 1) * (ARENA - 1));
}

function restart() {
  score = 0; timeLeft = ROUND_S; over = false;
  player.position.set(0, 0.5, 0);
  while (orbs.length < ORBS) { const orb = new THREE.Mesh(orbGeometry, orbMaterial); orbs.push(orb); scene.add(orb); }
  orbs.forEach(placeOrb);
}

function update(dt) {
  if (over) return;
  const x = (keys.has('KeyD') || keys.has('ArrowRight') ? 1 : 0) - (keys.has('KeyA') || keys.has('ArrowLeft') ? 1 : 0);
  const z = (keys.has('KeyS') || keys.has('ArrowDown') ? 1 : 0) - (keys.has('KeyW') || keys.has('ArrowUp') ? 1 : 0);
  const speed = 7;
  player.position.x = THREE.MathUtils.clamp(player.position.x + x * speed * dt, -ARENA + 0.5, ARENA - 0.5);
  player.position.z = THREE.MathUtils.clamp(player.position.z + z * speed * dt, -ARENA + 0.5, ARENA - 0.5);
  player.rotation.y += (x || z) ? dt * 4 : 0;
  for (const orb of orbs) {
    orb.position.y = 0.5 + Math.sin(performance.now() / 300 + orb.id) * 0.12;
    if (orb.position.distanceTo(player.position) < 0.9) { score += 1; placeOrb(orb); }
  }
  timeLeft -= dt;
  if (timeLeft <= 0) {
    over = true; timeLeft = 0;
    if (score > best) { best = score; localStorage.setItem('best', String(best)); }  // kept per app between sessions
  }
}

function render() {
  camera.position.lerp(new THREE.Vector3(player.position.x, 11, player.position.z + 9), 0.1);
  camera.lookAt(player.position);
  renderer.render(scene, camera);
  hud.score.textContent = `Score ${score}`;
  hud.best.textContent = `Best ${best}`;
  hud.message.textContent = over ? 'Time! Press R to play again' : `${Math.ceil(timeLeft)} s`;
}

addEventListener('resize', () => { camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix(); renderer.setSize(innerWidth, innerHeight); });

restart();
const timer = new THREE.Timer();
renderer.setAnimationLoop(time => {
  timer.update(time);
  update(Math.min(timer.getDelta(), 0.05));  // clamped step: no jump after a pause
  render();
});

window.wangpState = () => ({ score, best, timeLeft: +timeLeft.toFixed(1), over, player: [+player.position.x.toFixed(2), +player.position.z.toFixed(2)] });
