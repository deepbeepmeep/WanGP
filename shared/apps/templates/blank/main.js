// The app's code. Libraries come from the import map (three, three/addons/..., cannon-es, phaser, @tweenjs/tween.js, simplex-noise, tone);
// files of the app load with relative paths (assets/...). localStorage keeps data per app. Keep window.wangpState() returning the key values.
const root = document.getElementById('app');
let clicks = 0;
root.textContent = 'Ready';
root.addEventListener('click', () => { clicks += 1; root.textContent = `Clicked ${clicks} time${clicks > 1 ? 's' : ''}`; });

window.wangpState = () => ({ clicks });
