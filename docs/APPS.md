# Apps

Deepy Prime can build small web apps for you: interactive animations, games, explorable three.js scenes, physics toys and music visualizers. You play them inside WanGP, in the chat while Deepy works on them, and later from the Apps gallery. Their pictures, textures, 3D models and music can be made by WanGP's own models, so an app can feature the characters, worlds and songs you generated.

## Turn it on

Web apps are an advanced option, off by default:

1. Use **Deepy Prime** with the **Python Sandbox** enabled (**Configuration > Prompt Enhancer / Deepy > Python**).
2. Set **App Development (Advanced)** to **Web Apps** in the same tab, then save. Deepy starts a new conversation with its app tools, and the **Apps** gallery appears beside the other galleries.

Set it back to **Off** to hide apps again: Deepy no longer offers to build them, apps no longer start, and the Apps gallery is hidden. The app files stay where they are.

## Ask for an app

Describe what you want to play or explore, as you would to a game developer:

- "Make a small 3D game where a fox collects glowing berries in a forest clearing; use generated textures."
- "Turn my last song into a rhythm game: notes fall on the beat, arrow keys to hit them."
- "Build a solar system I can orbit, with planets textured by Flux and labels I can toggle."
- "Make a generative screensaver from this palette."

Deepy creates the app from a starting template, generates the assets it needs, writes the code, and runs the app in a hidden browser to check it: errors, frame rate, whether the score rises when keys are pressed, what the screen looks like over a few seconds. It fixes what it finds before showing it to you.

## Play and give feedback

Each time Deepy changes an app, the chat shows a card with a **Play** button. Play opens the app player beside the chat:

- The controls of the app are listed under it (for example **WASD** move, **Space** jump). Click inside the app first so that it receives your keys.
- Drag the player by its title bar to move it anywhere in the window; it stays fully visible.
- **Fill the Window** and **Full Screen** enlarge it. **Minimize** pauses the app and shrinks the player to its title bar, which you can park anywhere while you chat; **Restore** brings the app back.
- **Pause** stops its animation and sound. Apps also pause while hidden, outside the active view, or when WanGP loses focus. Showing the player again resumes the same app where it stopped. **Close**, **Reload**, or opening another app ends that session.
- The viewport menu shows how the app looks at desktop, tablet and phone widths.
- **Console** lists what the app reported (a red count shows errors).
- **Tell Deepy What Happened** starts a message summarizing the errors and frame rate; finish it with what you saw ("the fox falls through the floor after the bridge"). Deepy can also read the full report of what happened while you played.

When Deepy changes the app you are playing, the player reloads it.

## Keep your apps

When an app works, ask Deepy to publish it. It joins the **Apps** gallery as a `.wgpapp` file in your outputs, with its thumbnail, name, version and controls. Select it and press **Play** to open it again at any time, without Deepy. Each publication is a new version; ask Deepy to continue an app and publish again to keep improving it. This works in a later conversation too, and for an app someone gave you: Deepy reopens it from the Apps gallery, and its next publication gets the next version number.

Apps keep their own saved data, such as best scores, progress or settings, from one session to the next and across new versions.

The download button of the player saves the `.wgpapp` file so that you can share it. **Import** in Media Info adds a `.wgpapp` received from someone else to your Apps gallery.

## Previews of notes and tables

Even without web apps, Markdown notes, tables (CSV), JSON data and SVG drawings that Deepy saves as deliverables open as previews in the chat, rendered without running any code. With web apps on, HTML pages open the same way as apps.

## Safety

Apps are code written by an AI, possibly after reading web pages, so WanGP keeps them apart from everything else:

- An app runs in your browser in an isolated frame. It cannot use WanGP's controls (generation, files, galleries, settings), cannot read your WanGP session, and cannot reach the internet or your local network: everything it loads comes from its own folder and from libraries bundled with WanGP.
- It cannot open other windows or move the WanGP page to another address.
- Its saved data stays with that app.
- Apps never run on WanGP's server or in Python; Deepy's Python sandbox only builds them, with its own limits.

Like any web page, an app uses your graphics card while it plays: pause or minimize it during long generations.

---

> Applies to: Web apps built by Deepy Prime (App Development (Advanced) set to Web Apps): asking for apps, playing them in the app player, the Apps gallery, publishing, sharing and importing .wgpapp files, saved app data, chat previews of notes and tables, and the isolation of apps.
