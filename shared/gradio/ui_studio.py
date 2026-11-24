# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""WanGP's shared visual styling, preserving component and event definitions."""
import re
from pathlib import Path
from urllib.parse import quote

import gradio as gr

css_path = Path(__file__).with_name('ui_studio.css')
THEME_CHOICES = [('Blue Sky (Default)', 'default'), ('Classic Gradio', 'gradio'), ('Emerald', 'emerald'), ('Amethyst', 'amethyst'), ('Rose', 'rose')]
AZURE = gr.themes.colors.Color(name='azure', c50='#f1f7fc', c100='#d9ebfa', c200='#c2ddf5', c300='#91bce2', c400='#69a8df', c500='#357fc4', c600='#2b6ba8', c700='#205786', c800='#19466d', c900='#153956', c950='#10283e')
THEME_HUES = {'default': AZURE, 'emerald': 'emerald', 'amethyst': 'violet', 'rose': 'rose'}


def create_theme(name, **sizes):
    if name == 'gradio':
        return None
    theme = gr.themes.Soft(font=[gr.themes.Font('Verdana')], primary_hue=THEME_HUES[name], secondary_hue=AZURE if name == 'default' else 'indigo', neutral_hue='slate', **sizes)
    if name == 'default':
        theme.set(background_fill_primary='*neutral_50', slider_color_dark='*primary_400', checkbox_label_background_fill_selected='*primary_600')
    if name != 'default':
        theme.name = name
    return theme


icons = {
    'Apply': '<path d="m5 12 4 4L19 6"/>',
    'Refresh': '<path d="M20 7v5h-5M4 17v-5h5M6 7a7 7 0 0 1 12-1l2 6M4 12l2 6a7 7 0 0 0 12-1"/>',
    'Save': '<path d="M5 3h12l4 4v14H3V3h2Zm2 0v7h10V3M7 21v-7h10v7"/>',
    'Delete': '<path d="M3 6h18M9 6V3h6v3M5 6l1 15h12l1-15M10 10v7M14 10v7"/>',
    'Generate': '<path d="m8 4 12 8-12 8Z"/>',
    'Set Settings as Default': '<path d="m5 12 4 4L19 6"/>',
    'Export Settings to File': '<path d="M10 3H4v18h15v-7M14 3h7v7M11 13 21 3"/>',
    'Reset Settings': '<path d="M4 8a9 9 0 1 1-1 7M4 3v5h5M9 11h7M9 16h7M11 9v4M14 14v4"/>',
    'Enhance Prompt': '<path d="m15 4 1.8 5.2L22 11l-5.2 1.8L15 18l-1.8-5.2L8 11l5.2-1.8ZM5 2v6M2 5h6M5 16v6M2 19h6"/>',
    'Go Ahead Save it !': '<path d="m5 12 4 4L19 6"/>',
    'Go Ahead Delete it !': '<path d="M3 6h18M9 6V3h6v3M5 6l1 15h12l1-15M10 10v7M14 10v7"/>',
    "Don't do it !": '<path d="m6 6 12 12M6 18 18 6"/>',
    'Download': '<path d="M12 3v12m-5-5 5 5 5-5M4 16v5h16v-5"/>',
    'Exit': '<path d="M9 4H4v16h5M8 12h13m-5-5 5 5-5 5"/>',
    'Abort': '<rect x="5" y="5" width="14" height="14" rx="2"/>',
    'One More': '<rect x="3" y="3" width="12" height="14" rx="2"/><path d="M7 21h10a2 2 0 0 0 2-2v-4M9 7v6M6 10h6"/>',
    'Extend': '<path d="M4 5v14M4 12h16m-5-5 5 5-5 5"/>',
    'Pause': '<rect x="5" y="4" width="4" height="16" rx="1"/><rect x="15" y="4" width="4" height="16" rx="1"/>',
    'Resume': '<path d="m8 4 12 8-12 8Z"/>',
    'Early Stop': '<path d="m4 5 10 7-10 7Z"/><path d="M19 5v14"/>',
    'Add New Prompt To Queue': '<path d="M4 5h11M4 10h11M4 15h6M17 13v8m-4-4h8"/>',
    'Check Cache Size': '<path d="M3 17a9 9 0 1 1 18 0H3Zm9-5 4-4M5 12l2 1m5-8v2m7 5-2 1"/><circle cx="12" cy="13" r="1"/>',
}
icons['Clear Cache Now'] = icons['Delete']
icon_css = []
for index, (label, paths) in enumerate(icons.items()):
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="black" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">{paths}</svg>'
    icon_css.append(f'.wangp-studio-icon-{index} {{ --studio-icon: url("data:image/svg+xml,{quote(svg)}"); }}')


def style_config(config, theme):
    components = {component['id']: component for component in config['components']}
    paths = {}
    def visit(node, parents=()):
        paths[node['id']] = (*parents, node['id'])
        for child in node.get('children', []):
            visit(child, paths[node['id']])
    visit(config['layout'])
    scope = config['layout']['id']
    navigation = {next(node for node in reversed(paths[c['id']]) if components[node]['type'] == 'row') for c in components.values() if c['props'].get('elem_id') == 'family_list'}
    kinds = {kind: 'field' for kind in ('textbox', 'dropdown', 'slider', 'number', 'checkbox', 'checkboxgroup', 'radio', 'hierarchyselector', 'rangeslider', 'file', 'image', 'audio')}
    kinds.update(button='button', uploadbutton='button', downloadbutton='button', accordion='accordion', tabs='tabs', gallery='gallery', video='gallery')
    studio_renderer = getattr(gr.routes.FileResponse, '_wangp_frontend', False)
    native_targets = {}
    for component_id, component in components.items():
        if component_id not in paths or scope not in paths[component_id]:
            continue
        extra = ['wangp-studio-dropdown'] if component['type'] == 'dropdown' else []
        in_navigation = bool(navigation.intersection(paths[component_id]))
        if not in_navigation and component['type'] in kinds:
            extra.append('wangp-studio-' + kinds[component['type']])
        if not in_navigation and component['type'] == 'button':
            # Models can rename the prompt enhancer (Write, Magic Prompt, etc.).
            label = 'Enhance Prompt' if 'btn_centered' in (component['props'].get('elem_classes') or []) else component['props']['value']
            if label in icons:
                extra.extend(['wangp-studio-icon', f'wangp-studio-icon-{list(icons).index(label)}'])
            if label in ('Generate', 'Add New Prompt To Queue'):
                extra.append('wangp-studio-generation-action')
            if label == 'Enhance Prompt':
                extra.append('wangp-studio-enhancer')
        if extra and studio_renderer:
            component['wangp_studio_classes'] = extra
        elif extra:
            element_id = component['props'].get('elem_id') or f'component-{component_id}'
            for role in extra:
                native_targets.setdefault(role, {})[element_id] = None
    scope_selector = f'#component-{scope}'
    css = css_path.read_text(encoding='utf8')
    if isinstance(theme, gr.themes.Default):
        css += '\n' + css_path.with_name('ui_studio_classic.css').read_text(encoding='utf8')
    elif theme.name in ('emerald', 'amethyst', 'rose'):
        css += '\n' + css_path.with_name(f'ui_studio_{theme.name}.css').read_text(encoding='utf8')
    css += '\n' + '\n'.join(icon_css)
    css = css.replace('.wangp-studio-scope', scope_selector)
    if studio_renderer:
        # The scope ID supplies the same specificity as the old component ID.
        css = re.sub(r'\.wangp-studio-[a-z0-9-]+', lambda m: f'{scope_selector} :where({m.group()})', css)
    else:
        # A standalone Gradio preview has no WanGP renderer. Expand each
        # role-headed selector in the theme to ordinary, individually indexed
        # ID selectors. Plugin elem_classes stay entirely application-owned.
        def native_selector(match):
            role, suffix = match.groups()
            suffix = suffix.rstrip()
            continues = suffix.endswith(',')
            if continues:
                suffix = suffix[:-1]
            selectors = (f':where({scope_selector}) #{element_id}{suffix}' for element_id in native_targets.get(role, ()))
            return (', '.join(selectors) or ':not(*)') + (',' if continues else ' ')
        css = re.sub(r'(?m)^[ \t]*\.(wangp-studio-[a-z0-9-]+)([^\n{]*)', native_selector, css)
    config['css'] = (config['css'] or '') + '\n' + css
    return config


class Blocks(gr.Blocks):
    def get_config_file(self):
        return style_config(super().get_config_file(), self.theme)
