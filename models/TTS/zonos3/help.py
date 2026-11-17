# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
DEFAULT_PROMPT = "Hello there. It's lovely to finally meet you. I've heard so much about your work, and I'd love to hear how the new project is going."

INFOS = """ZONOS3 Mini turns text into 44.1 kHz speech in 43 languages, with one of 27 built-in voices or a voice cloned from a short recording, and lets you steer its emotion, delivery and voice. It is a small, fast model (about 1.8 GB): it runs on modest GPUs and generates speech much faster than real time.

### Voices

- **Built-in Voice** speaks with the voice selected in **Voice**. Labels give each voice's language and style (Irish, British, energetic, ASMR, announcer...). A voice keeps its accent in other languages: pick one of the target language for native pronunciation.
- **Voice Reference** clones the voice of a recording: use a clean recording of one person, 10 to 30 seconds long (at most 60 seconds are used).

### Emotions and Delivery

- **Cues in the text** change the emotion or the manner from where they are written: `[happy] What a day! [sad] But it is over now.` A cue lasts until the next cue; in a dialogue, each line starts again from the Voice Controls. The available cues are listed in the prompt help.
- The **Presets** (in the Presets menu above the prompt) fill **Voice Controls** with ready-made combinations: personas (BBC Newsreader, Gravelly Old Man, Late-night Radio...) and uses (Audiobook Narrator, Bedtime Story, Sports Commentator...). They keep your voice, text and other settings; Neutral (Reset) clears them.
- **Voice Controls** set the starting delivery of the whole speech and the voice design. Edit a preset's values or write your own: styles such as `happy` or `whisper`, or controls such as `pace=-1, deep=2` (see the prompt help).
- A cue replaces the emotion of the Voice Controls, and any control it names; the rest of the Voice Controls stays. `[neutral]` goes back to the Voice Controls.
- Voice design controls (gender, age, pitch, textures, room, accents) change the voice itself and only go in Voice Controls: they cannot change in the middle of the text. Built-in voices then switch to their speaker fingerprint, which keeps the voice but less of its recording's delivery.
- A style pushes fully for its first 4 seconds, then holds at about a third of its strength so it does not drift on long passages. Pace and whisper stay constant.
- A cue in the middle of the text starts as the voice reaches its words, with a short fade. The new delivery can take a word or two to settle, and a whisper or anger may linger briefly after the next cue: give each delivery at least a sentence.

### Dialogues

Write each line as `Speaker 1: ...`, `Speaker 2: ...`. With built-in voices, Speaker 1 uses **Voice** and **Other Speakers' Voices** can name the voices of Speakers 2 and 3 (for example `Rupert, Willow`); speakers left without a voice take the next voices of the Voice list, those of Speaker 1's language first, so that each speaker sounds different. With **Voice References for Speakers 1 and 2** (or **1, 2 and 3**), each speaker uses its own recording. Each line is generated on its own with its speaker's voice and the lines are joined with **Pause between Multi Speakers sentences**: speakers do not interrupt or overlap each other. Voice Controls apply to every speaker; start a line with a cue to give it its own emotion.

### Settings

- **Language** selects how numbers, dates, abbreviations and symbols are read ("$5.32" becomes "five dollars thirty two cents"). It does not change the voice or its accent.
- **Speaking Rate** asks for a slower or faster delivery; Auto lets the voice and the text decide. `[slow]` and `[fast]` cues change the pace of part of the text.
- **Maximum Speech Duration** caps the speech (the whole dialogue). Speech usually ends sooner, when the text has been read; if the end of the text is missing, raise it.
- Temperature 0.75, Top-k 80 and Top-p 0.69 are the released defaults. A lower temperature gives a steadier, flatter delivery; a higher one a livelier but less reliable one. Generate again with another seed for another take.
"""

PROMPT_INFOS = """Write the words to speak, in their language, with natural punctuation: commas, full stops, question and exclamation marks shape the rhythm and intonation.

- Numbers, dates, currencies and abbreviations are read in full in the selected **Language**: `On 12/03, Dr. Smith paid $5.32.`
- `[pause]` inserts a pause of about half a second after the word before it; `[pause 1s]` or `[pause 2.5s]` sets its length: `Wait for it... [pause 1s] there it is.`

### Cues

A cue in square brackets changes the delivery from where it is written until the next cue (in a dialogue, until the end of its line). No closing tag is needed.

- **Styles:** `[happy]`, `[sad]`, `[angry]`, `[calm]`, `[scared]`, `[confused]`, `[excited]`, `[gloomy]`, `[whisper]`, `[loud]`, `[chatty]`, `[reading]` (careful reading), `[lively]`, `[flat]`, `[hesitant]`, `[slow]`, `[fast]`; `[neutral]` goes back to the Voice Controls.
- **Strength:** add a number, 1 being the tested strength and 2 the maximum: `[happy 1.5]`, `[whisper 0.5]`. A cue in the middle of the text can sound subtler than the same style from the start: raise it if needed (`[loud 1.5]`).
- **Combinations:** separate them with commas: `[sad, whisper]`, `[excited, fast]`.
- **Exact values** of the emotion and delivery controls listed below: `[pace=-1.5]`, `[happy=3, projection=-1]`.
- Cues are a fixed set: free descriptions (`[warm and relaxed]`) and sounds (`[laughs]`, `[sigh]`) are refused before generation, and so are voice design controls, which only go in Voice Controls.

```text
[calm] The house was quiet that night. [scared] Then something moved downstairs. [whisper] Did you hear that?
```

For a dialogue, start each line with its speaker, and with a cue to set the line's emotion:

```text
Speaker 1: [excited] Did you see the message from the station?
Speaker 2: [calm] I did. They want us there before dawn.
Speaker 1: [whisper] Then we leave tonight. [pause] Pack light.
```

### Voice Controls

Comma-separated items for the whole speech: styles as in cues (`happy`, `sad 1.5`) and `name=value` controls, for example `happy=3, pace=-1` or `gender=-1, age=2, raspy=3, deep=2`. Values beyond the tested range are clipped; the usual range is shown after each name.

- **Emotion** (any voice, also in cues): `happy` (0-4), `sad` (0-6), `angry` (0-4), `calm` (0-6), `scared` (0-12), `confused` (0-3), `valence` (-4 negative to 4 positive), `arousal` (-4 calm to 4 excited).
- **Delivery** (any voice, also in cues): `pace` (-2 slower to 2 faster), `projection` (-1.5 whisper to 2 loud), `chatty` (-4 careful reading to 4 spontaneous), `lively` (-8 flat to 8 lively), `hesitant` (-8 fluent to 8 hesitant), `emphasis` (0-6).
- **Voice design** (changes the voice, Voice Controls only): `gender` (-2 masculine to 2 feminine), `age` (-2 younger to 2 older), `pitch` (-2 to 2), `range` (-2 monotone to 2 varied intonation), `deep`, `bright`, `husky`, `raspy`, `soft`, `authority` (0-4 each), `room` (-2 phone line to 4 clean studio).
- **Accents** (changes the voice, Voice Controls only, 0-1.5): `accent:american`, `accent:british`, `accent:scottish`, `accent:irish`, `accent:welsh`, `accent:australian`, `accent:indian`, and foreign accents in English `accent:hin`, `accent:spa`, `accent:fra`, `accent:por`, `accent:deu`, `accent:vie`, `accent:pol`, `accent:mandarin`.
- `conditioning=embedding` gives a built-in voice its speaker fingerprint instead of its recording (freer delivery), `conditioning=prefix` forces the recording (voice design controls then refuse); `normalize=off` reads the text exactly as written.
"""

DEEPY_INFOS = """ZONOS3 Mini: fast multilingual TTS (43 languages), 44.1 kHz, steerable. `audio_prompt_type` "" = built-in voice `custom_settings.voice` (27 voice ids, e.g. "Imogen_en-British_F_1"; labels give language and style); "A" = clone `audio_guide` (clean single-speaker recording, 10-30 s) through a speaker embedding; "AB"/"ABD" = dialogue with `audio_guide2`/`audio_guide3`. Dialogue lines `Speaker N:`; each line is generated separately with its speaker's voice, joined with `pause_seconds`; with built-in voices `custom_settings.dialogue_voices` may name Speakers 2-3 (e.g. "Rupert, Willow"); unnamed speakers get distinct voices automatically (next voices of the list, same language first). Other `custom_settings`: `language` (text-normalization code, default "en_us"; not the accent), `speaking_rate` ("auto", "very_slow" ... "very_fast"). Emotion and delivery: inline cues in the prompt (see prompt guide), and `alt_prompt` = Voice Controls for the whole speech, comma-separated styles (`happy`, `sad 1.5`) and `name=value`: emotion happy 0-4, sad 0-6, angry 0-4, calm 0-6, scared 0-12, confused 0-3, valence/arousal -4..4; delivery pace -2..2, projection -1.5 (whisper)..2 (loud), chatty -4..4, lively -8..8, hesitant -8..8, emphasis 0-6; voice design (alt_prompt only; moves the voice, switches built-in voices to embedding conditioning) gender -2 masc..2 fem, age -2..2, pitch -2..2, range -2..2, deep/bright/husky/raspy/soft/authority 0-4, room -2 phone..4 studio, accent:<american|british|scottish|irish|welsh|australian|indian|hin|spa|fra|por|deu|vie|pol|mandarin> 0-1.5; options conditioning=embedding|prefix, normalize=off. A cue replaces the alt_prompt emotion and the controls it names; `[neutral]` returns to alt_prompt. Presets fill `alt_prompt`. `duration_seconds` is a maximum. Sampling defaults: temperature 0.75, top_k 80, top_p 0.69."""

DEEPY_PROMPT_INFOS = """Words to speak in their language, naturally punctuated; numbers and abbreviations are read in full per `custom_settings.language`. `[pause]` (0.5 s) or `[pause 1.5s]` after a word inserts a pause. Cues in square brackets change the delivery from where they stand until the next cue (in a dialogue, the end of the `Speaker N:` line), no closing tag: styles `[happy]`, `[sad]`, `[angry]`, `[calm]`, `[scared]`, `[confused]`, `[excited]`, `[gloomy]`, `[whisper]`, `[loud]`, `[chatty]`, `[reading]`, `[lively]`, `[flat]`, `[hesitant]`, `[slow]`, `[fast]`, `[neutral]` (back to alt_prompt); optional strength 0-2 (`[happy 1.5]`), combinations (`[sad, whisper]`), exact emotion/delivery controls (`[pace=-1, happy=3]`). Any other bracket (free descriptions, sounds like `[laughs]`, voice design) is refused. A mid-text cue starts as the voice reaches its words and needs a word or two to settle: keep each delivery for at least a sentence. Dialogue: one `Speaker N:` line per turn, e.g. `Speaker 2: [calm] I did.`"""

_CUE_RULES = """Direct the delivery with cues in square brackets, which the voice performs instead of speaking. Only these exact cue words exist, always written in English whatever the script's language: [happy], [sad], [angry], [calm], [scared], [confused], [excited], [gloomy], [whisper], [loud], [chatty], [reading] (careful reading), [lively], [flat], [hesitant], [slow], [fast], and [neutral] to go back to the plain delivery.
- A cue changes the delivery from where it is written until the next cue; there is no closing tag.
- Put a cue right before the sentence or phrase where the delivery changes, and keep each delivery for at least a whole sentence: a new delivery needs a few words to settle. Do not repeat the cue already in effect.
- Two cues may share one bracket, separated by a comma: [sad, whisper], [excited, fast]. A strength may follow a cue word, 1 being normal and 2 the maximum: [happy 1.5]. Use strengths sparingly, only for a clearly stronger emotion.
- [pause] adds about half a second of silence after the word before it; [pause 1s] sets its length. Keep it for dramatic beats and let punctuation make the rhythm otherwise (commas, "...", dashes).
- Any other word in square brackets is an error that stops the generation. For an emotion with no cue word, use the closest listed cue or a pair of them: impatient or eager [excited, fast], defensive or irritated [angry], annoyed [angry 0.5], cautious or gentle [calm], tender [calm, whisper], shy or unsure [hesitant], bored [flat], tired [gloomy, slow], cheerful [happy], surprised [excited], nervous [scared, hesitant], mysterious [whisper], proud [loud], thoughtful [reading, slow].
- Never write free descriptions such as [warm and relaxed], sounds or non-verbal tags such as [laughs] or [sigh], directions in parentheses, asterisks, angle-bracket tags or emojis: they are refused or read aloud. Convey laughter and hesitation with words and punctuation."""
_CUE_CHECK = "Before answering, check every bracket: the only words allowed inside square brackets are happy, sad, angry, calm, scared, confused, excited, gloomy, whisper, loud, chatty, reading, lively, flat, hesitant, slow, fast, neutral and pause (with an optional strength or pause length). Replace any other word, even a common emotion such as defensive, impatient, nervous or surprised, with the closest allowed cue."

ENHANCER = f"""Write only the speech script of one speaker for ZONOS3 Mini, a text-to-speech model. Preserve the user's language, exact supplied wording, meaning and explicit exclusions; if the user supplied the exact words to speak, keep them word for word and only add cues. If asked to write speech from a brief, write the finished words to speak, 3 to 8 sentences unless another length is asked, in natural spoken language with clear punctuation; write numbers, dates and abbreviations normally, they are read in full.
{_CUE_RULES}
Start the script with a cue when its first sentence is not plain neutral speech, and let the delivery follow the content: most sentences need no new cue. Keep the cues and [pause] tags the user already wrote. Do not add speaker labels, titles, explanations, Markdown or commentary. {_CUE_CHECK} Return only the script. The example shows the format only: write new content for the user's request.

Example:
[calm] The house was quiet that night, and nobody had spoken for hours. [scared] Then something moved downstairs, slowly, step by step. [whisper] Did you hear that? [pause] Don't move."""

DIALOGUE_ENHANCER = f"""Write a dialogue between two people for ZONOS3 Mini, a text-to-speech model, from the user prompt; write three people only if the user asks for three (at most three). Preserve the user's language, exact supplied wording, characters, meaning and explicit exclusions; if the user supplied exact lines, keep them word for word and only add cues. Write 6 to 14 lines unless another length is asked, in natural spoken language with clear punctuation. Every line starts with its speaker label, Speaker 1: or Speaker 2: (Speaker 3: for a third person), one turn per line, each character always keeping the same number.
Each line is spoken as a separate take that starts again from the plain delivery: begin a line with a cue whenever its delivery is not plain neutral speech, even when the same speaker's previous line had the same cue. A cue later in a line changes the delivery for the rest of that line.
{_CUE_RULES}
Keep the cues, [pause] tags and speaker labels the user already wrote. Do not add a narrator, character names before the labels, stage directions, explanations, Markdown or commentary. {_CUE_CHECK} Return only the dialogue lines. The example shows the format only: write new content for the user's request.

Example:
Speaker 1: [excited] Did you see the message from the station?
Speaker 2: [calm] I did. They want us there before dawn.
Speaker 1: [confused] Before dawn? Why so early? You never told me anything!
Speaker 2: [angry, fast] Because you never listen when I do!
Speaker 1: [hesitant] Fine... fine. [calm] Tell me now.
Speaker 2: [whisper] Someone is watching the roads. [pause] We leave tonight.
Speaker 1: [scared, slow] Then I had better pack light."""
