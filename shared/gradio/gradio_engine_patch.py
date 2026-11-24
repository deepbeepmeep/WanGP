# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Cache Gradio constructor bookkeeping and safely traverse shared state."""
import inspect
import os
import shutil
import tempfile
from functools import lru_cache, wraps
from pathlib import Path

from shared.utils.media_settings import MediaSettings


def _updateable(fn):
    from gradio.component_meta import get_local_contexts

    # Gradio 5.29 inspects the same constructor on every component/update.
    fn_args = inspect.getfullargspec(fn).args

    @wraps(fn)
    def wrapper(*args, **kwargs):
        self = args[0]
        initialized_before = hasattr(self, '_constructor_args')
        if not initialized_before:
            self._constructor_args = []
        for i, arg in enumerate(args):
            if i == 0 or i >= len(fn_args):
                continue
            kwargs[fn_args[i]] = arg
        self._constructor_args.append(kwargs)
        in_event_listener, is_render = get_local_contexts()
        if in_event_listener and initialized_before and not is_render:
            return None
        return fn(self, **kwargs)

    return wrapper


@lru_cache(maxsize=8)
def _resolved_upload_folder(temp_dir):
    return str((Path(temp_dir) / 'gradio').resolve())


def _get_upload_folder():
    # Preserve explicit overrides, including relative paths, verbatim. Key the
    # default by its absolute base so tempfile.tempdir/cwd changes invalidate it.
    return os.environ.get('GRADIO_TEMP_DIR') or _resolved_upload_folder(os.path.abspath(tempfile.gettempdir()))


def _move_uploaded_files_to_cache(files, destinations):
    for source, destination in zip(files, destinations):
        # Upload destinations include the content hash. On Windows rename fails
        # when one already exists; shutil.move would then truncate the cached
        # file while browsers may still be reading it with its original length.
        if Path(destination).exists():
            Path(source).unlink()
        else:
            shutil.move(source, destination)


def _snapshot_traverse(value, func, is_root):
    if is_root(value):
        return func(value)
    if isinstance(value, dict):
        # gr.State can retain dictionaries shared with a background worker.
        # Snapshot entries before recursion/file I/O can let that worker resize
        # the mapping. Keep the original leaves and file-cache processing.
        return {key: _snapshot_traverse(item, func, is_root) for key, item in list(value.items())}
    if isinstance(value, MediaSettings):  # entries not read yet hold no file data: reading every one would load each gallery item's metadata
        return value.map_loaded(lambda item: _snapshot_traverse(item, func, is_root))
    if isinstance(value, (list, tuple)):
        return [_snapshot_traverse(item, func, is_root) for item in value]
    return value


def _same_constructor_args(kwargs, current):
    if kwargs.keys() != current.keys():
        return False
    try:
        return all(kwargs[key] is current[key] or kwargs[key] == current[key] for key in kwargs)
    except Exception:
        return False


# Gradio 5.29's Blocks.postprocess_data constructs a new component for every
# update dictionary so the session holds the component the update describes.
# A form refresh sends hundreds of updates whose arguments equal the session
# component's; constructing those copies dominated postprocessing. Reuse the
# component when the arguments are unchanged; everything else is the original.
_POSTPROCESS_DATA_HASH = 'b9962fc450b9ccb8'
_STATE_IDS_HASH = '74d61935b157993c'


async def _postprocess_data(self, block_fn, predictions, state):
    from gradio import blocks, components, processing_utils, utils

    state = state or blocks.SessionState(self)
    if isinstance(predictions, dict) and predictions == blocks.skip() and len(block_fn.outputs) > 1:
        predictions = [blocks.skip()] * len(block_fn.outputs)
    if isinstance(predictions, dict) and len(predictions) > 0:
        predictions = blocks.convert_component_dict_to_list([block._id for block in block_fn.outputs], predictions)
    if len(block_fn.outputs) == 1 and not block_fn.batch:
        predictions = [predictions]
    self.validate_outputs(block_fn, predictions)
    output = []
    for i, block in enumerate(block_fn.outputs):
        try:
            if predictions[i] is components._Keywords.FINISHED_ITERATING:
                output.append(None)
                continue
        except (IndexError, KeyError) as err:
            raise ValueError(f"Number of output components does not match number of values returned from from function {block_fn.name}") from err
        if block.stateful:
            if not utils.is_prop_update(predictions[i]):
                state[block._id] = predictions[i]
            output.append(None)
            continue
        prediction_value = predictions[i]
        if utils.is_prop_update(prediction_value):
            prediction_value = utils.delete_none(prediction_value, skip_value=True)
        if isinstance(prediction_value, blocks.Block):
            prediction_value = prediction_value.constructor_args.copy()
            prediction_value["__type__"] = "update"
        if utils.is_prop_update(prediction_value):
            current = state[block._id]
            kwargs = current.constructor_args.copy()
            kwargs.update(prediction_value)
            kwargs.pop("value", None)
            kwargs.pop("__type__")
            kwargs["render"] = False
            if not _same_constructor_args(kwargs, current.constructor_args):
                state[block._id] = block.__class__(**kwargs)
            prediction_value = blocks.postprocess_update_dict(block=state[block._id], update_dict=prediction_value, postprocess=block_fn.postprocess)
        elif block_fn.postprocess:
            if not isinstance(block, components.Component):
                raise blocks.InvalidComponentError(f"{block.__class__} Component not a valid output component.")
            if block._id in state:
                block = state[block._id]
            prediction_value = block.postprocess(prediction_value)
            if isinstance(prediction_value, (blocks.GradioModel, blocks.GradioRootModel)):
                prediction_value_serialized = prediction_value.model_dump()
            else:
                prediction_value_serialized = prediction_value
            prediction_value_serialized = await processing_utils.async_move_files_to_cache(prediction_value_serialized, block, postprocess=True)
            if block._id not in state:
                state[block._id] = block
            state[block._id].value = prediction_value_serialized
        output.append(await processing_utils.async_move_files_to_cache(prediction_value, block, postprocess=True))
    return output


def _get_state_ids_to_track(self, block_fn, state):
    from gradio import utils

    if state is None:
        return [], []
    state_ids_to_track, hashed_values = [], []
    stateful = [block for block in block_fn.outputs if block.stateful]
    if stateful:
        # One pass over the listeners instead of one per stateful output.
        changed = {target for fn in state.blocks_config.fns.values() for target in fn.targets}
        for block in stateful:
            if (block._id, "change") in changed:
                state_ids_to_track.append(block._id)
                hashed_values.append(utils.deep_hash(state[block._id]))
    return state_ids_to_track, hashed_values


def _install_postprocess_patch(blocks):
    import hashlib
    for name, expected in (('postprocess_data', _POSTPROCESS_DATA_HASH), ('get_state_ids_to_track', _STATE_IDS_HASH)):
        source = inspect.getsource(getattr(blocks.Blocks, name))
        if hashlib.sha256(source.encode()).hexdigest()[:16] != expected:
            raise RuntimeError(f'Gradio engine patch requires the pinned Gradio 5.29 Blocks.{name}')
    blocks.Blocks.postprocess_data = _postprocess_data
    blocks.Blocks.get_state_ids_to_track = _get_state_ids_to_track


def install():
    from gradio import blocks, component_meta, data_classes, processing_utils, routes, utils
    from gradio_client import utils as client_utils
    from shared.gradio import gradio_frontend_patch, gradio_image_preprocess_cache, gradio_media_response_patch, gradio_model_change_queue, gradio_queue_wakeup_patch

    gradio_queue_wakeup_patch.install()
    gradio_image_preprocess_cache.install()
    gradio_model_change_queue.install()
    gradio_media_response_patch.install()
    gradio_frontend_patch.install()
    routes.move_uploaded_files_to_cache = _move_uploaded_files_to_cache
    if blocks.Blocks.postprocess_data is not _postprocess_data:
        _install_postprocess_patch(blocks)

    if component_meta.updateable is _updateable:
        return
    original = component_meta.updateable
    wrapper_code = original(lambda self: None).__code__
    # Most built-ins are imported before WanGP installs its patches. Rewrap only
    # Gradio's own updateable wrappers, leaving plugin constructor decorators intact.
    pending, visited = [blocks.Block], set()
    while pending:
        cls = pending.pop()
        if cls in visited:
            continue
        visited.add(cls)
        pending.extend(cls.__subclasses__())
        constructor = cls.__dict__.get('__init__')
        if inspect.isfunction(constructor) and constructor.__code__ is wrapper_code:
            cls.__init__ = _updateable(constructor.__wrapped__)
    component_meta.updateable = _updateable
    blocks.get_upload_folder = utils.get_upload_folder = processing_utils.get_upload_folder = _get_upload_folder
    client_utils.traverse = data_classes.traverse = _snapshot_traverse
