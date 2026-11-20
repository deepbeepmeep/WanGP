# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Small, deduplicated UI events that need media metadata rather than pixels."""
from functools import wraps

import gradio as gr
from gradio import blocks


def bind(triggers, sources, callback, state, outputs):
    """Sources are (component, value/count/presence/editor/gallery) pairs."""
    @wraps(callback)
    def refresh(state_value, metadata):
        return callback(state_value, metadata)

    refresh._wangp_metadata_inputs = [(component._id, kind) for component, kind in sources]
    return gr.on(triggers=triggers, fn=refresh, inputs=[state, gr.JSON(visible=False)], outputs=outputs, queue=False, trigger_mode="always_last", show_progress="hidden", api_name=False)


def install():
    original = blocks.Blocks.get_config_file
    if getattr(original, "_wangp_metadata_events", False):
        return

    @wraps(original)
    def get_config_file(self):
        config = original(self)
        for dependency in config["dependencies"]:
            sources = getattr(self.fns[dependency["id"]].fn, "_wangp_metadata_inputs", None)
            # Only bind() supplies this list. A plugin callable can implement
            # __getattr__ and return a proxy for an unknown attribute.
            if isinstance(sources, list):
                dependency["wangp_metadata_inputs"] = sources
        return config

    get_config_file._wangp_metadata_events = True
    blocks.Blocks.get_config_file = get_config_file


# All maps belong to the mounted Blocks instance, so pages and embedded apps
# never share a signature. A later event replaces only an unsent snapshot.
PREPARE_JS = """
let wangpMetadata;
if(R.wangp_metadata_inputs){
    const token={};
    wangpMetadataPending.set(S,token);
    await new Promise(resolve=>setTimeout(resolve,0));
    if(wangpMetadataPending.get(S)!==token)return;
    wangpMetadataPending.delete(S);
    wangpMetadata=R.wangp_metadata_inputs.map(([id,kind])=>P(id,kind));
    const signature=JSON.stringify(wangpMetadata);
    if(wangpMetadataSent.get(S)===signature)return;
    wangpMetadataSent.set(S,signature);
}
"""

READ_JS = """
if(g&&wangpKind){
    const value=g.props.value;
    if(wangpKind==="value")return value;
    if(wangpKind==="count")return value?.length||0;
    if(wangpKind==="editor")return g.instance.get_metadata();
    if(wangpKind==="gallery")return {items:value?.map(item=>(item.image||item.video).path)||[],selected:g.props.selected_index};
    return value&&( !Array.isArray(value)||value.length) ? "present" : null;
}
"""
