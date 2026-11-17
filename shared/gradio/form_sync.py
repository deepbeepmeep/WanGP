# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Bind existing Gradio components and save callbacks to a shared form."""
import json

import gradio as gr
from gradio.blocks import Block, convert_component_dict_to_list

from shared.utils.form_sync import FormConflict, Saved


class GradioForm:
    def __init__(self, registry, name, components, *, read=None, scope=lambda: 'application', keys=None, view_events=(), render_fields=None, lock=None):
        self.components = list(components)
        self.keys = keys or [str(component._id) for component in components]
        self.labels = {key: component.label or key for key, component in zip(self.keys, components)}
        initial = dict(zip(self.keys, [component.value for component in components]))
        self.form = registry.register(name, initial, read=read, scope=scope, labels=self.labels, lock=lock)
        self.view_events = list({(event.fn, tuple(event.inputs), tuple(event.outputs)): event for event in view_events}.values())
        self.render_fields = render_fields
        self.view_outputs = list(dict.fromkeys(output for event in self.view_events for output in event.outputs if output not in self.components))
        self.baseline = gr.Textbox(value=json.dumps({'scope': scope(), 'revision': -1, 'values': initial}), visible=False)
        self.remote = gr.Textbox(visible=False)
        self.notice = gr.HTML()
        with gr.Row(visible=False) as self.conflict_controls:
            reload = gr.Button('Load saved values', size='sm')
            keep = gr.Button('Keep my edits', size='sm')
        trigger_id = f'wangp-form-sync-{name}'
        selector = f'[id={json.dumps(trigger_id)}] button, button[id={json.dumps(trigger_id)}]'
        trigger = gr.Button(visible=False, elem_id=trigger_id)
        keys_json, labels_json = json.dumps(self.keys), json.dumps(self.labels)
        self.apply_js = f'(remote, baseline, ...values) => [JSON.stringify(window.WanGPFormSync.apply({keys_json}, {labels_json}, remote, baseline, values))]'
        # Gradio supplies output values to JS after the inputs. Read the live
        # baseline/fields there; Python receives exactly one serialized update.
        self.apply_inputs = [self.remote]
        self.apply_outputs = [self.baseline, *self.components, *self.view_outputs, self.notice, self.conflict_controls]
        event = trigger.click(self.snapshot, outputs=[self.remote], queue=False, show_progress='hidden', trigger_mode='always_last', api_name=False)
        self._bind_apply(event)
        self._bind_apply(reload.click(self.snapshot, outputs=[self.remote], queue=False, show_progress='hidden', api_name=False), js=f'(remote, baseline, ...values) => [JSON.stringify(window.WanGPFormSync.apply({keys_json}, {labels_json}, remote, baseline, values, true))]')
        keep.click(self.snapshot, outputs=[self.remote], queue=False, show_progress='hidden', api_name=False).then(fn=None, inputs=[self.remote], outputs=[self.baseline, self.notice, self.conflict_controls], js='remote => [remote, "", {__type__:"update",visible:false}]')
        root = gr.context.Context.root_block
        root.load(fn=None, js=f'() => {{window.WanGPFormSync.register({json.dumps(name)}, {json.dumps(selector)});document.querySelector({json.dumps(selector)})?.click();}}')

    def _bind_apply(self, event, js=None):
        def apply_updates(payload):
            # JS merges live drafts; Python postprocessing must also update the
            # session's component choices before its next input is validated.
            return tuple(json.loads(payload))
        return event.then(apply_updates, inputs=self.apply_inputs, outputs=self.apply_outputs, js=js or self.apply_js, preprocess=False, queue=False, show_progress='hidden', api_name=False)

    def snapshot(self):
        snapshot = self.form.snapshot()
        values = {component: snapshot['values'][key] for component, key in zip(self.components, self.keys)}
        updates = {component: gr.update() for component in [*self.components, *self.view_outputs]}
        dependencies = {component: set() for component in updates}
        keys = dict(zip(self.components, self.keys))
        for event in self.view_events:
            args = [values.get(component, component.value) for component in event.inputs]
            result = event.fn(*args)
            # Callbacks also return component-keyed dictionaries or component
            # instances. Normalize them as Gradio does before serializing.
            if isinstance(result, dict) and result:
                if result == gr.skip() and len(event.outputs) > 1:
                    result = [gr.skip() for _ in event.outputs]
                else:
                    result = convert_component_dict_to_list([component._id for component in event.outputs], result)
            results = [result] if len(event.outputs) == 1 else result
            for component, result in zip(event.outputs, results):
                if isinstance(result, Block):
                    result = {**result.constructor_args, '__type__': 'update'}
                update = result if isinstance(result, dict) and result.get('__type__') == 'update' else gr.update(value=result)
                if component in updates:
                    updates[component].update(update)
                    dependencies[component].update(keys[item] for item in event.inputs if item in keys)
        snapshot['view'] = [{'update': updates[c], 'dependencies': sorted(dependencies[c])} for c in self.view_outputs]
        snapshot['fields'] = {key: {'update': {k: v for k, v in updates[c].items() if k != 'value'}, 'dependencies': sorted(dependencies[c])} for c, key in keys.items()}
        if self.render_fields:
            for key, update in self.render_fields(snapshot['values']).items():
                snapshot['fields'][key]['update'].update({k: v for k, v in update.items() if k != 'value'})
        return json.dumps(snapshot)

    def bind_save(self, trigger, save, *, inputs, outputs, fields=None, **event_options):
        indices = [inputs.index(component) for component in self.components]

        def submit(baseline, *args):
            local = {key: args[index] for key, index in zip(self.keys, indices)}
            baseline = json.loads(baseline)
            submitted = local
            if fields is not None:
                current = self.form.snapshot()['values']
                submitted = {**current, **{key: local[key] for key in fields}}
                baseline = {**baseline, 'values': {**current, **{key: baseline['values'][key] for key in fields}}}
            def commit(values):
                merged = list(args)
                for key, index in zip(self.keys, indices):
                    merged[index] = values[key]
                return save(*merged)
            try:
                result = self.form.submit(baseline, submitted, commit)
            except FormConflict as error:
                gr.Info(str(error))
                payload = self.snapshot()
                return (*((gr.update(),) * len(outputs)), payload) if outputs else payload
            snapshot = json.loads(self.snapshot())
            if isinstance(result, Saved):
                snapshot['ack'] = local if fields is None else {key: local[key] for key in fields}
            payload = json.dumps(snapshot)
            return (*(result.output if isinstance(result, Saved) else result), payload) if outputs else payload

        event = trigger(submit, inputs=[self.baseline, *inputs], outputs=[*outputs, self.remote], **event_options)
        return self._bind_apply(event)
