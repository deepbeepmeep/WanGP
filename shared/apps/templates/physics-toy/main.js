import * as THREE from 'three';
import * as CANNON from 'cannon-es';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// three.js draws, cannon-es simulates: each body has a mesh copied from it after every physics step.
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(innerWidth, innerHeight);
renderer.shadowMap.enabled = true;
document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x111622);
const camera = new THREE.PerspectiveCamera(50, innerWidth / innerHeight, 0.1, 200);
camera.position.set(9, 7, 12);
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 2, 0);
scene.add(new THREE.HemisphereLight(0xdde8ff, 0x22283a, 1.0));
const sun = new THREE.DirectionalLight(0xffffff, 2.4);
sun.position.set(6, 14, 8);
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
Object.assign(sun.shadow.camera, { left: -12, right: 12, top: 12, bottom: -12 });
scene.add(sun);

const world = new CANNON.World({ gravity: new CANNON.Vec3(0, -9.82, 0) });
world.allowSleep = true;
const ground = new CANNON.Body({ type: CANNON.Body.STATIC, shape: new CANNON.Plane() });
ground.quaternion.setFromEuler(-Math.PI / 2, 0, 0);
world.addBody(ground);
const floor = new THREE.Mesh(new THREE.PlaneGeometry(60, 60), new THREE.MeshStandardMaterial({ color: 0x27304a, roughness: 0.95 }));
floor.rotation.x = -Math.PI / 2;
floor.receiveShadow = true;
scene.add(floor);

const pairs = [];
function add(mesh, body) {
  mesh.castShadow = mesh.receiveShadow = true;
  scene.add(mesh); world.addBody(body); pairs.push({ mesh, body });
}
function clear() {
  for (const { mesh, body } of pairs) { scene.remove(mesh); world.removeBody(body); }
  pairs.length = 0;
}
function buildTower() {
  clear();
  const size = 0.9, colors = [0xf28482, 0xf6bd60, 0x84a59d, 0x7cc4ff];
  for (let level = 0; level < 8; level++) for (let column = -1; column <= 1; column++) {
    const body = new CANNON.Body({ mass: 1, shape: new CANNON.Box(new CANNON.Vec3(size / 2, size / 2, size / 2)), position: new CANNON.Vec3(column * size * 1.02, size / 2 + level * size, 0) });
    add(new THREE.Mesh(new THREE.BoxGeometry(size, size, size), new THREE.MeshStandardMaterial({ color: colors[(level + column + 4) % colors.length] })), body);
  }
}

let thrown = 0, pointer = null;
const raycaster = new THREE.Raycaster();
renderer.domElement.addEventListener('pointerdown', event => { pointer = { x: event.clientX, y: event.clientY }; });
renderer.domElement.addEventListener('pointerup', event => {
  if (!pointer || Math.hypot(event.clientX - pointer.x, event.clientY - pointer.y) > 6) return;  // a drag orbits, a click throws
  raycaster.setFromCamera(new THREE.Vector2(event.clientX / innerWidth * 2 - 1, -(event.clientY / innerHeight) * 2 + 1), camera);
  const direction = raycaster.ray.direction;
  const body = new CANNON.Body({ mass: 4, shape: new CANNON.Sphere(0.4), position: new CANNON.Vec3().copy(camera.position) });
  body.velocity.set(direction.x * 28, direction.y * 28, direction.z * 28);
  add(new THREE.Mesh(new THREE.SphereGeometry(0.4, 24, 16), new THREE.MeshStandardMaterial({ color: 0xffffff, metalness: 0.4, roughness: 0.3 })), body);
  thrown += 1;
});
addEventListener('keydown', event => { if (event.code === 'KeyR') { buildTower(); thrown = 0; } });
addEventListener('resize', () => { camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix(); renderer.setSize(innerWidth, innerHeight); });

buildTower();
const timer = new THREE.Timer();
renderer.setAnimationLoop(time => {
  timer.update(time);
  world.step(1 / 60, Math.min(timer.getDelta(), 0.05), 3);  // fixed physics step, clamped frame time
  for (const { mesh, body } of pairs) { mesh.position.copy(body.position); mesh.quaternion.copy(body.quaternion); }
  controls.update();
  renderer.render(scene, camera);
});

window.wangpState = () => ({ bodies: pairs.length, thrown, standing: pairs.filter(({ body }) => body.mass === 1 && body.position.y > 1).length, awake: pairs.filter(({ body }) => body.sleepState === CANNON.Body.AWAKE).length });
