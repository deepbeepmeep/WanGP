import Phaser from 'phaser';

// A 2D platformer scene with arcade physics. Its textures are drawn in code so that it runs without assets; pictures made with WanGP load in
// preload(): this.load.image('hero', 'assets/hero.webp'), this.load.spritesheet('run', 'assets/run.png', { frameWidth: 64, frameHeight: 64 }),
// this.load.audio('music', 'assets/music.ogg').
class Play extends Phaser.Scene {
  constructor() { super('play'); }

  create() {
    const brush = this.add.graphics();
    brush.fillStyle(0x4cc9f0).fillRoundedRect(0, 0, 40, 48, 8); brush.generateTexture('hero', 40, 48); brush.clear();
    brush.fillStyle(0x3a4a6b).fillRect(0, 0, 64, 20); brush.generateTexture('ground', 64, 20); brush.clear();
    brush.fillStyle(0xffd166).fillCircle(12, 12, 12); brush.generateTexture('coin', 24, 24); brush.destroy();

    this.platforms = this.physics.add.staticGroup();
    for (let x = 32; x < 1280; x += 64) this.platforms.create(x, 710, 'ground');
    for (const [x, y] of [[300, 560], [620, 440], [940, 330], [420, 260]]) for (let index = 0; index < 3; index++) this.platforms.create(x + index * 64, y, 'ground');

    this.player = this.physics.add.sprite(120, 600, 'hero').setCollideWorldBounds(true).setBounce(0.05);
    this.physics.add.collider(this.player, this.platforms);
    this.coins = this.physics.add.group({ key: 'coin', repeat: 9, setXY: { x: 80, y: 0, stepX: 120 } });
    this.coins.children.iterate(coin => { coin.setBounceY(Phaser.Math.FloatBetween(0.3, 0.6)); });
    this.physics.add.collider(this.coins, this.platforms);
    this.physics.add.overlap(this.player, this.coins, (player, coin) => {
      coin.disableBody(true, true);
      this.score += 1;
      this.scoreText.setText(`Score ${this.score}`);
      if (this.coins.countActive(true) === 0) this.coins.children.iterate(item => { item.enableBody(true, item.x, 0, true, true); });
    });
    this.score = 0;
    this.scoreText = this.add.text(16, 16, 'Score 0', { fontFamily: 'system-ui, sans-serif', fontSize: '28px', color: '#f2f5f8' });
    this.keys = this.input.keyboard.addKeys('LEFT,RIGHT,UP,SPACE,R');
  }

  update() {
    const { LEFT, RIGHT, UP, SPACE, R } = this.keys;
    if (Phaser.Input.Keyboard.JustDown(R)) { this.scene.restart(); return; }
    this.player.setVelocityX(LEFT.isDown ? -260 : RIGHT.isDown ? 260 : 0);
    if ((UP.isDown || SPACE.isDown) && this.player.body.blocked.down) this.player.setVelocityY(-560);
  }
}

const game = new Phaser.Game({
  type: Phaser.AUTO, parent: 'game', backgroundColor: '#10131a', width: 1280, height: 720,
  scale: { mode: Phaser.Scale.FIT, autoCenter: Phaser.Scale.CENTER_BOTH },  // fits the window at any size, keeping 16:9
  physics: { default: 'arcade', arcade: { gravity: { y: 1100 } } },
  scene: [Play],
});

window.wangpState = () => {
  const scene = game.scene.getScene('play');
  if (!scene || !scene.player) return { loading: true };
  return { score: scene.score, player: [Math.round(scene.player.x), Math.round(scene.player.y)], onGround: scene.player.body.blocked.down, coinsLeft: scene.coins.countActive(true) };
};
