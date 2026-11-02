// A 2D canvas loop without a library: sharp at any pixel density, sized to the window, with a clamped time step.
// Pictures made with WanGP load as images: const sprite = new Image(); sprite.src = 'assets/sprite.webp';
const canvas = document.getElementById('view');
const context = canvas.getContext('2d');
let width = 0, height = 0;

function resize() {
  const ratio = Math.min(devicePixelRatio, 2);
  width = innerWidth; height = innerHeight;
  canvas.width = Math.round(width * ratio); canvas.height = Math.round(height * ratio);
  context.setTransform(ratio, 0, 0, ratio, 0, 0);  // draw in CSS pixels
}
addEventListener('resize', resize);
resize();

const particles = [];
let frozen = false, bursts = 0;

function burst(x, y, count = 60) {
  bursts += 1;
  for (let index = 0; index < count; index++) {
    const angle = Math.random() * Math.PI * 2, speed = 60 + Math.random() * 240;
    particles.push({ x, y, vx: Math.cos(angle) * speed, vy: Math.sin(angle) * speed, life: 1.6 + Math.random(), hue: (bursts * 47 + index) % 360 });
  }
}

canvas.addEventListener('pointerdown', event => burst(event.clientX, event.clientY));
addEventListener('keydown', event => { if (event.code === 'Space') { frozen = !frozen; event.preventDefault(); } });

function update(dt) {
  if (frozen) return;
  for (const particle of particles) {
    particle.vy += 160 * dt;
    particle.x += particle.vx * dt; particle.y += particle.vy * dt;
    particle.life -= dt;
  }
  for (let index = particles.length - 1; index >= 0; index--) if (particles[index].life <= 0) particles.splice(index, 1);
  if (particles.length < 40) burst(width * (0.2 + Math.random() * 0.6), height * (0.25 + Math.random() * 0.3));
}

function draw() {
  context.fillStyle = 'rgba(13, 15, 20, 0.25)';  // trails
  context.fillRect(0, 0, width, height);
  for (const particle of particles) {
    context.fillStyle = `hsla(${particle.hue}, 90%, 62%, ${Math.min(1, particle.life)})`;
    context.beginPath();
    context.arc(particle.x, particle.y, 3, 0, Math.PI * 2);
    context.fill();
  }
}

let last = performance.now();
function frame(now) {
  update(Math.min((now - last) / 1000, 0.05));  // clamped: no jump after a pause
  last = now;
  draw();
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);

window.wangpState = () => ({ particles: particles.length, bursts, frozen, size: [width, height] });
