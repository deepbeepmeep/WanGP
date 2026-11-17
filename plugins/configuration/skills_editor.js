// Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
(() => {
    const escape = text => text.replace(/[&<>]/g, char => ({"&": "&amp;", "<": "&lt;", ">": "&gt;"})[char]);
    const token = (kind, text) => `<span class="skill-${kind}">${escape(text)}</span>`;

    function inline(text) {
        // Preserve every source character; only colour spans, never render user HTML.
        const pattern = /(`+)([^`\n]+)\1|\*\*[^*\n]+\*\*|__[^_\n]+__|\*[^*\n]+\*|_[^_\n]+_|~~[^~\n]+~~|!?\[[^\]\n]*\]\([^\n)]*\)/g;
        let result = "", start = 0;
        for (const match of text.matchAll(pattern)) {
            const value = match[0];
            const kind = value.startsWith("`") ? "code" : /^!?\[/.test(value) ? "link" : /^(\*\*|__)/.test(value) ? "strong" : "em";
            result += escape(text.slice(start, match.index)) + token(kind, value);
            start = match.index + value.length;
        }
        return result + escape(text.slice(start));
    }

    function highlight(text) {
        let yaml = false, fence = null;
        return text.split("\n").map((line, index) => {
            if (index === 0 && line === "---") { yaml = true; return token("marker", line); }
            if (yaml) {
                if (/^(---|\.\.\.)\s*$/.test(line)) { yaml = false; return token("marker", line); }
                const key = line.match(/^(\s*[\w-]+:)(.*)$/);
                return key ? token("key", key[1]) + escape(key[2]) : escape(line);
            }
            const marker = line.match(/^ {0,3}(`{3,}|~{3,})(.*)$/);
            if (fence) {
                if (marker && marker[1][0] === fence[0] && marker[1].length >= fence.length && !marker[2].trim()) fence = null;
                return token("code", line);
            }
            if (marker) { fence = marker[1]; return token("code", line); }
            if (/^ {0,3}(#{1,6})(\s|$)/.test(line) || /^ {0,3}(=+|-+)\s*$/.test(line)) return token("heading", line);
            const prefix = line.match(/^(\s*(?:>\s*|[-+*]\s+|\d+[.)]\s+))(.*)$/);
            return prefix ? token("marker", prefix[1]) + inline(prefix[2]) : inline(line);
        }).join("\n") + "\n";
    }

    function update(text) {
        const root = window.gradioApp ? window.gradioApp() : document;
        const field = root.querySelector("#deepy_skill_content textarea");
        if (!field) return;
        let layer = field.parentElement.querySelector(".skill-highlight");
        if (!layer) {
            layer = document.createElement("pre");
            layer.className = "skill-highlight";
            layer.setAttribute("aria-hidden", "true");
            field.before(layer);
            field.classList.remove("scroll-hide");
            field.classList.add("skill-highlighted");
            field.spellcheck = false;
            field.setAttribute("aria-label", "SKILL.md");
            field.addEventListener("scroll", () => {
                layer.scrollTop = field.scrollTop;
                layer.scrollLeft = field.scrollLeft;
            }, {passive: true});
        }
        layer.innerHTML = highlight(text || "");
        layer.scrollTop = field.scrollTop;
        layer.scrollLeft = field.scrollLeft;
    }
    window.WanGPSkillEditor = {update};
})();
