import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// Renderer filling the window (the page never scrolls)
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(innerWidth, innerHeight);
document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0b1020);
const camera = new THREE.PerspectiveCamera(50, innerWidth / innerHeight, 0.1, 200);
camera.position.set(4, 3, 6);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.target.set(0, 1.2, 0);

// Light: without it standard materials render black
scene.add(new THREE.HemisphereLight(0xbfd8ff, 0x202030, 1.2));
const sun = new THREE.DirectionalLight(0xffffff, 2.5);
sun.position.set(5, 8, 4);
scene.add(sun);

const ground = new THREE.Mesh(new THREE.CircleGeometry(8, 64), new THREE.MeshStandardMaterial({ color: 0x1c2638, roughness: 0.9 }));
ground.rotation.x = -Math.PI / 2;
scene.add(ground);

// Replace with the app's own content. Assets load with paths relative to the app folder:
//   new THREE.TextureLoader().load('assets/texture.webp')
//   new GLTFLoader().load('assets/model.glb', gltf => scene.add(gltf.scene))   (import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js')
const subject = new THREE.Mesh(new THREE.TorusKnotGeometry(1, 0.32, 160, 24), new THREE.MeshStandardMaterial({ color: 0x7cc4ff, metalness: 0.3, roughness: 0.35 }));
subject.position.y = 1.6;
scene.add(subject);

addEventListener('resize', () => {
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
});

const timer = new THREE.Timer();
let elapsed = 0;
renderer.setAnimationLoop(time => {
  timer.update(time);
  const dt = Math.min(timer.getDelta(), 0.05);  // clamped: after a pause the scene continues instead of jumping
  elapsed += dt;
  subject.rotation.y += dt * 0.6;
  subject.position.y = 1.6 + Math.sin(elapsed * 1.5) * 0.15;
  controls.update();
  renderer.render(scene, camera);
});

// The app's key values, read by app_check after each step: keep it up to date with what matters in this app
window.wangpState = () => ({
  elapsed: +elapsed.toFixed(2),
  rotation: +subject.rotation.y.toFixed(2),
  camera: camera.position.toArray().map(value => +value.toFixed(2)),
  objects: scene.children.length,
});
