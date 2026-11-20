# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Serialize browser model-selection requests through their form refresh."""
from functools import wraps

from gradio import blocks
from gradio.events import EventListenerMethod


# These hooks target the pinned Gradio 5.29 frontend. Queue captured requests,
# not DOM clicks: select event data and the user's requested filter stay intact.
_QUEUE_JS = """
const wangpModelQueues = new Map();
function wangpModelEnqueue(group, run) {
    let queue = wangpModelQueues.get(group);
    if (!queue) wangpModelQueues.set(group, queue = []);
    queue.push(run);
    if (queue.length === 1) run();
}
function wangpModelRelease(group) {
    const queue = wangpModelQueues.get(group);
    if (!queue) return;
    queue.shift();
    if (queue.length) queue[0]();
    else wangpModelQueues.delete(group);
}
"""
_REPLACEMENTS = {
    'function Jt(S,J=null,K=null){': _QUEUE_JS + 'function Jt(S,J=null,K=null){',
    'if(pe===void 0)return;const R=pe;': 'if(pe===void 0)return;if(pe.wangp_model_end){wangpModelRelease(S);return}const R=pe;',
    'async function Qt(W,Ie=!1){': 'async function Qt(W,Ie=!1,wangpAdmitted=false){if(R.wangp_model_root&&!wangpAdmitted){wangpModelEnqueue(R.wangp_model_root[0],()=>Qt(W,Ie,true));return}',
    'pl(Ee,ne),Mn(n)}function Bo': 'pl(Ee,ne),Mn(n);if(R.wangp_model_root){const [ack,index]=R.wangp_model_root,value=Ee[index];if(value?.__type__==="update"&&!("value"in value))Jt(ack)}}function Bo',
    'else if(ne.stage==="error"){': 'else if(ne.stage==="error"){if(R.wangp_model_root)Jt(R.wangp_model_root[0]);',
    'catch(ae){if(d.closed)return;': 'catch(ae){if(R.wangp_model_root)Jt(R.wangp_model_root[0]);if(d.closed)return;',
}


def _prepare(app):
    if not hasattr(app, '_wangp_model_switch_acks'):
        app._wangp_model_switch_acks = {}
    for target in app.blocks.values():
        if target.elem_id != 'wangp_model_choice_target' or target._id in app._wangp_model_switch_acks:
            continue
        # Plugins may add model-change listeners and branch existing chains.
        # Release after WanGP's form refresh, independently of those listeners.
        tails = [fn for fn in app.fns.values() if getattr(fn.fn, '_wangp_model_switch_tail', False) is True and target in fn.inputs]
        if tails:
            [tail] = tails
        else:
            [change] = [fn for fn in app.fns.values() if fn.name == 'change_model_from_target' and target in fn.inputs]
            [tail] = [fn for fn in app.fns.values() if fn.name == 'fill_inputs' and fn.trigger_after == change._id]
        # Gradio dispatches this frontend-only continuation after its pending
        # output updates settle. It adds no HTTP request or wait to a single swap.
        _, ack = app.default_config.set_event_trigger([EventListenerMethod(None, 'then')], None, None, None, js='()=>{}', trigger_after=tail._id, queue=False, api_name=False)
        app._wangp_model_switch_acks[target._id] = ack


def install():
    original = blocks.Blocks.get_config_file
    if getattr(original, '_wangp_model_queue', False):
        return

    @wraps(original)
    def get_config_file(self):
        _prepare(self)
        config = original(self)
        for dependency in config['dependencies']:
            for target, ack in self._wangp_model_switch_acks.items():
                if dependency['id'] == ack:
                    dependency['wangp_model_end'] = True
                elif dependency['trigger_after'] is None and target in dependency['outputs']:
                    dependency['wangp_model_root'] = [ack, dependency['outputs'].index(target)]
        return config

    get_config_file._wangp_model_queue = True
    blocks.Blocks.get_config_file = get_config_file
