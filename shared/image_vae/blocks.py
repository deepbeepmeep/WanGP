# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""The blocks of GroupNorm image VAEs (AutoencoderKL: FLUX, FLUX 2, Z-Image, SD3, SD), whatever the names their classes give the layers:
GroupNorm + SiLU + 3x3 convolution pairs with a shortcut, the x2 upsampling convolution, the output convolution.

Under CUDA autocast to a 16-bit dtype they run by strips of rows on channels-last activations (shared.kernels.vae_ops.image_norm_conv:
GroupNorm statistics of the whole image in FP32, the normalized rows rounded once, each convolution writing its rows in place), with the
block that halves the channels keeping its hidden state and its shortcut in the rows of its own input. Elsewhere (no autocast, CPU, small
activations) the same layers run whole, as PyTorch computes them. The module wrappers stay with each model; nothing here names a model.

VAEs whose norms are per pixel (an RMS norm over the channels: the Wan architecture and its image descendants) have nothing global in a
residual block: pixel_resnet_block, pixel_norm_conv and add_channel_shortcut_ compute them by bands of rows with only the block's input
and output at full size. Rows goes further: an activation whose rows are computed when its reader asks for them (ResnetRows,
UpsampledRows, NormConvRows), so that blocks, upsamplings and the output convolution chain band after band and none of their
activations exists whole; bands() hands the last one's rows to the caller, a band at a time."""
import torch
import torch.nn.functional as F

from shared.kernels import vae_ops
from shared.utils.conv_bands import BAND_BYTES, band_rows, conv_bands


def norm_swish(norm, x):
    """swish(norm(x)) for a convolution. Under autocast, GroupNorm computes in float32 and the convolution casts its input to the autocast dtype:
    the shared operation rounds once to that dtype without float32 tensors in full (vae_ops.group_norm_silu)."""
    if not torch.is_autocast_enabled(x.device.type):
        return vae_ops.silu_(norm(x))
    return vae_ops.group_norm_silu(x, norm, torch.get_autocast_dtype(x.device.type))


def resnet_block(x_list, norm1, conv1, norm2, conv2, shortcut=None, scratch=None):
    """conv2(swish(norm2(conv1(swish(norm1(x)))))) + shortcut(x) (shortcut None: x itself). Consumes [x]. scratch, when supplied, is a list
    that borrows the disjoint hidden half of this level's row-interleaved storage, reused by the following equal-width blocks."""
    x = x_list.pop()
    cin, cout = conv1.in_channels, conv1.out_channels
    if vae_ops.image_norm_conv_supported(x, norm1):
        if x.stride(1) != 1:
            x = x.contiguous(memory_format=torch.channels_last)
        if cin == 2 * cout and x.is_contiguous(memory_format=torch.channels_last) and vae_ops.image_norm_conv_is_banded(x):
            # Each original NHWC row becomes two shorter NHWC rows: conv1's hidden state and the shortcut. The last row
            # waits for the next strip to read its halo. No full-size normalized input or additional residual is needed.
            n, _, height, width = x.shape
            shape = (n, cout, height, width)
            strides = (x.stride(0), 1, x.stride(2), cout)
            hidden = x.as_strided(shape, strides)
            residual = x.as_strided(shape, strides, x.storage_offset() + width * cout)
            pending = None
            for start, stop, band in vae_ops.image_norm_conv_bands(x, norm1, conv1):
                projected = shortcut(x[:, :, start:stop])
                if pending is not None:
                    row, h_row, r_row = pending
                    hidden[:, :, row:row+1].copy_(h_row)
                    residual[:, :, row:row+1].copy_(r_row)
                    pending = None
                    del h_row, r_row
                end = stop if stop == height else stop - 1
                if end < stop:
                    pending = (end, band[:, :, -1:].clone(), projected[:, :, -1:].clone())
                hidden[:, :, start:end].copy_(band[:, :, :end-start])
                residual[:, :, start:end].copy_(projected[:, :, :end-start])
                del band, projected
            if scratch is not None:
                scratch[:] = [hidden]
            x = residual
            h = hidden
        else:
            if shortcut is not None and scratch is not None:
                scratch.clear()
            h = vae_ops.image_norm_conv([x], norm1, conv1, out=scratch[0] if scratch else None)
            if shortcut is not None:
                x = shortcut(x)
        h_list, h = [h], None
        return vae_ops.image_norm_conv(h_list, norm2, conv2, out=x, add=True)
    if scratch is not None:
        scratch.clear()
    x = x.contiguous()  # small PyTorch norms use NCHW; avoid their FP32 channels-last conversion
    h = norm_swish(norm1, x)
    if shortcut is not None:
        x = shortcut(x)
    h = conv1(h)
    h = norm_swish(norm2, h)
    h = conv2(h)
    return h.add_(x) if h.dtype == x.dtype else x + h  # in h, this block's own tensor; a 32-bit input keeps its 32-bit sum


def upsample(x_list, conv):
    """conv of the nearest x2 upsampling of x, by rows: no full-size upsampled input. Consumes [x]."""
    x = x_list.pop()
    if torch.is_autocast_enabled(x.device.type) and x.is_cuda:
        # Convert before expansion, and release the spent hidden half of a row-interleaved workspace.
        x = x.contiguous(memory_format=torch.channels_last)
    return conv_bands(x, conv, upsample=True)


def norm_conv(x_list, norm, conv):
    """conv(swish(norm(x))), the output convolution of an encoder or a decoder. Consumes [x]."""
    if vae_ops.image_norm_conv_supported(x_list[0], norm):
        return vae_ops.image_norm_conv(x_list, norm, conv)
    return conv(norm_swish(norm, x_list.pop()))


CHAIN_MIN_PIXELS = 8192  # a band of fewer pixels falls below the size from which the row-window convolution is used (its MIN_PIXELS)
CHAIN_MIN_BANDS = 16  # an activation of fewer bands (512 MB) stays whole: chaining it saves little and, without that convolution, costs time


def _read(x, lo, hi):
    return x.rows(lo, hi) if isinstance(x, Rows) else x[:, :, lo:hi]


class _ResnetBands:
    """A residual block (ops: norm_act1, conv1, norm_act2, conv2, shortcut) on x (a tensor or Rows) of `height` rows, a band of output
    rows at a time. Bands read down the image compute each row of the block's inner activations once: the last two rows of the
    normalized input and of the first convolution's normalized output stay for the next band, whose second convolution and first
    convolution read them (a band elsewhere recomputes its halo). Each value has the inputs of the whole-image computation."""

    def __init__(self, x, height, ops):
        self.x, self.height, self.ops, self.next = x, height, ops, None

    def band(self, first, last, destination):
        """Rows first:last, added over the shortcut in the tensor and from the row that destination(hidden rows) gives."""
        norm_act1, conv1, norm_act2, conv2, shortcut = self.ops
        height = self.height
        c_lo, c_hi = max(0, first - 1), min(height, last + 1)  # rows of the first convolution that the second one reads
        follows = self.next == first and first > 0
        c_new = self.c_hi if follows else c_lo  # those not computed yet
        a_lo, a_hi = max(0, c_new - 1), min(height, c_hi + 1)  # normalized input rows that they read
        a_new = self.a_hi if follows else a_lo
        base = min(first, a_new)
        rows = _read(self.x, base, a_hi)
        if a_new > a_lo:  # the rows kept from the band before, then the new ones
            a = torch.empty((*rows.shape[:2], a_hi - a_lo, rows.shape[3]), dtype=self.a.dtype, device=self.a.device, memory_format=torch.channels_last if self.a.stride(1) == 1 else torch.contiguous_format)
            a[:, :, :a_new - a_lo].copy_(self.a[:, :, a_lo - a_new:])
            if a_hi > a_new:
                norm_act1(rows[:, :, a_new - base:], out=a[:, :, a_new - a_lo:])
        else:
            a = norm_act1(rows[:, :, a_lo - base:])
        if c_new > c_lo:
            c = torch.empty((a.shape[0], conv1.out_channels, c_hi - c_lo, a.shape[3]), dtype=self.c.dtype, device=self.c.device, memory_format=torch.channels_last if self.c.stride(1) == 1 else torch.contiguous_format)
            c[:, :, :c_new - c_lo].copy_(self.c[:, :, c_lo - c_new:])
            conv1(a, window=(c_new - a_lo, c_hi - c_new), out=c, out_row=c_new - c_lo)
            norm_act2(c[:, :, c_new - c_lo:], out=c[:, :, c_new - c_lo:])
        else:
            c = conv1(a, window=(c_lo - a_lo, c_hi - c_lo))
            c = norm_act2(c, out=c)
        self.a, self.a_hi, self.c, self.c_hi, self.next = a[:, :, -2:].clone(), a_hi, c[:, :, -2:].clone(), c_hi, last
        del a
        out, row = destination(c)
        rows = rows[:, :, first - base:last - base]
        out[:, :, row:row + last - first].copy_(rows if shortcut is None else shortcut(rows))
        conv2(c, window=(first - c_lo, last - first), out=out, out_row=row, add=True)


def pixel_resnet_block(x, norm_act1, conv1, norm_act2, conv2, shortcut=None, band_bytes=BAND_BYTES):
    """conv2(norm_act2(conv1(norm_act1(x)))) + shortcut(x) (shortcut None: x itself) for x (batch, channels, height, width), where each
    norm_act is per pixel (silu(norm) of an RMS norm over the channels): every output row reads the input rows at most two away, so the
    block runs by bands of rows with only x and the result at full size, neither the normalized input nor the first convolution's output
    nor the shortcut, and each value has the inputs of the whole-image computation. x is borrowed; it may be Rows, computed as the
    bands read them.
    norm_act(rows, out=None): silu(norm(rows)) in a new tensor, or written into out (rows itself: in place). conv1, conv2: 3x3, stride 1,
    zero padding 1, called conv(rows, window=(first, count), out=None, out_row=0, add=False) as shared.utils.conv_bands.conv2d_weights
    (rows: the window's input rows and their halo where the image has one). shortcut(rows): a 1x1 convolution."""
    n, cin, height, width = x.shape
    cout = conv2.out_channels
    lazy = isinstance(x, Rows)
    row_bytes = n * max(cin, cout) * width * x.dtype.itemsize
    # PyTorch sums a norm over the channels of planes in an order that follows their shape (not so channels-last): a moderate input in
    # planes (the first block after a level too small for the row-window kernel) runs whole, and rounds as before the bands
    rows = height if not lazy and x.stride(1) != 1 and height * row_bytes <= 4 * band_bytes else band_rows(height, row_bytes, 1, band_bytes)
    out = []

    def destination(h):  # in the layout the convolutions work in (channels-last where the row-window kernel runs)
        if not out:
            out.append(torch.empty((n, cout, height, width), dtype=h.dtype, device=h.device, memory_format=torch.channels_last if h.stride(1) == 1 else torch.contiguous_format))
        return out[0], start

    block = _ResnetBands(x, height, (norm_act1, conv1, norm_act2, conv2, shortcut))
    for start in range(0, height, rows):
        block.band(start, min(height, start + rows), destination)
    return out[0]


def pixel_norm_conv(x, norm_act, conv, band_bytes=BAND_BYTES):
    """conv(norm_act(x)), the output convolution after a per-pixel norm (see pixel_resnet_block for norm_act and conv), by bands of
    rows: no full-size normalized input. x is borrowed."""
    n, cin, height, width = x.shape
    rows = band_rows(height, n * cin * width * x.element_size(), 1, band_bytes)
    if rows >= height:
        return conv(norm_act(x))
    out = None
    for start in range(0, height, rows):
        stop = min(height, start + rows)
        lo, hi = max(0, start - 1), min(height, stop + 1)
        band = conv(norm_act(x[:, :, lo:hi]), window=(start - lo, stop - start))
        if out is None:
            out = torch.empty((n, band.shape[1], height, width), dtype=band.dtype, device=band.device, memory_format=torch.channels_last if band.stride(1) == 1 else torch.contiguous_format)
        out[:, :, start:stop].copy_(band)
        del band
    return out


def channel_shortcut_indices(channels, repeats, step=4, first=0, device=None):
    """For add_channel_shortcut_: the source channel of each output channel, for the positions (0, 0), (0, 1), (1, 0), (1, 1)."""
    index = torch.arange(channels, device=device)
    return [(first + step * index + position) // repeats for position in range(4)]


def add_channel_shortcut_(x, source, repeats, step=4, first=0, row0=0, source_row0=0, band_bytes=BAND_BYTES, indices=None):
    """x += the x2 upsampling of source by channel duplication and depth-to-space: x (batch, c, 2h, 2w), source (batch, cs, h, w); the
    pixel (2i + a, 2j + b) of channel o receives the source channel (first + step o + 2 a + b) // repeats at (i, j): each source channel
    duplicated repeats times, then step of them per output channel, the four from `first` on going to the four positions (step 4, first
    0 for one frame; 8 and 4 for the second frame of a temporal x2). In place by bands of rows, one position at a time: neither the
    duplicated source nor its rearranged copy nor a new sum at full size. x may hold a window of the output's rows, from row0 on, and
    source the rows that window reads, from source_row0 on. indices: channel_shortcut_indices(...), for a caller that adds band
    after band. source is borrowed."""
    n, c, height, width = x.shape
    rows = band_rows(source.shape[2], n * c * source.shape[3] * x.element_size(), 0, band_bytes)
    indices = channel_shortcut_indices(c, repeats, step, first, x.device) if indices is None else indices
    for a in range(2):
        top = row0 + (a - row0) % 2  # the window's first output row of this parity
        for b in range(2):
            index = indices[2 * a + b]
            for start in range(top // 2, (row0 + height - a + 1) // 2, rows):
                stop = min((row0 + height - a + 1) // 2, start + rows)
                x[:, :, 2 * start + a - row0:2 * stop + a - row0:2, b::2].add_(source[:, :, start - source_row0:stop - source_row0].index_select(1, index))
    return x


class Rows:
    """An activation (batch, channels, height, width) that never exists whole: rows(lo, hi) returns its rows lo:hi, computing those it
    does not hold from the rows of its own input (a tensor, or Rows in turn: a chain). Reads go down the image. The last band read
    stays, with `retain` rows above it for a second reader a few rows behind (a shortcut), so that rows read again (a band's halo)
    are computed once. Subclasses give .shape and compute(band, offset, first, last): rows first:last written into band from its row
    offset. Inputs are borrowed while the object lives."""

    def __init__(self, shape, like, retain=0):
        self.shape, self.dtype, self.device, self.retain = tuple(shape), like.dtype, like.device, retain
        self.channels_last = like.channels_last if isinstance(like, Rows) else like.stride(1) == 1
        self.band, self.lo, self.hi = None, 0, 0

    def rows(self, lo, hi):
        if self.band is not None and self.lo <= lo and hi <= self.hi:
            return self.band[:, :, lo - self.lo:hi - self.lo]
        kept = self.band is not None and self.lo <= lo <= self.hi  # the band read before continues into this one
        start = max(self.lo, lo - self.retain) if kept else lo
        new = self.hi if kept else lo
        n, c, _, width = self.shape
        band = torch.empty((n, c, hi - start, width), dtype=self.dtype, device=self.device, memory_format=torch.channels_last if self.channels_last else torch.contiguous_format)
        if new > start:
            band[:, :, :new - start].copy_(self.band[:, :, start - self.lo:])
        self.band = None  # the band before is released before the new rows are computed
        self.compute(band, new - start, new, hi)
        self.band, self.lo, self.hi = band, start, hi
        return band[:, :, lo - start:]


class ResnetRows(Rows):
    """The residual block of pixel_resnet_block (same arguments) on x, a tensor or Rows, as rows computed when read."""

    def __init__(self, x, norm_act1, conv1, norm_act2, conv2, shortcut=None):
        super().__init__((x.shape[0], conv2.out_channels, *x.shape[2:]), x)
        self.block = _ResnetBands(x, x.shape[2], (norm_act1, conv1, norm_act2, conv2, shortcut))

    def compute(self, band, offset, first, last):
        self.block.band(first, last, lambda h: (band, offset))


class UpsampledRows(Rows):
    """conv(nearest x2 of x) + the channel shortcut of `source` (add_channel_shortcut_ with repeats, step, first; None: no shortcut), as
    rows computed when read: the x2 activation never exists whole. x and source: tensors or Rows (a source that is also x's own
    input follows x by a few rows: give it `retain`). conv: a convolution prepared by conv_bands.band_convs."""

    def __init__(self, x, conv, mode="nearest", source=None, repeats=1, step=4, first=0):
        super().__init__((x.shape[0], conv.out_channels, 2 * x.shape[2], 2 * x.shape[3]), x)
        self.x, self.conv, self.mode, self.shortcut = x, conv, mode, (source, repeats, step, first)
        self.indices = None if source is None else channel_shortcut_indices(conv.out_channels, repeats, step, first, self.device)

    def compute(self, band, offset, first, last):
        lo = max(0, (first - 1) // 2)
        self.conv(_read(self.x, lo, min(self.x.shape[2], last // 2 + 1)), upsample=True, mode=self.mode, window=(first - 2 * lo, last - first), out=band, out_row=offset)
        source, repeats, step, channel = self.shortcut
        if source is not None:
            add_channel_shortcut_(band[:, :, offset:], _read(source, first // 2, (last + 1) // 2), repeats, step, channel, row0=first, source_row0=first // 2, indices=self.indices)


class ConvRows(Rows):
    """conv(x) for a 3x3 convolution (pixel_resnet_block's conv interface) on x, a tensor or Rows, as rows computed when read: an
    encoder's first convolution."""

    def __init__(self, x, conv):
        super().__init__((x.shape[0], conv.out_channels, *x.shape[2:]), x)
        self.x, self.conv = x, conv

    def compute(self, band, offset, first, last):
        lo = max(0, first - 1)
        self.conv(_read(self.x, lo, min(self.shape[2], last + 1)), window=(first - lo, last - first), out=band, out_row=offset)


class DownsampledRows(Rows):
    """conv (3x3, stride 2, no padding of its own) of x zero-padded on its right and bottom, + shortcut(rows of source) (source None:
    no shortcut; shortcut maps the source rows 2i and 2i + 1 to row i: an average of pixel groups), as rows computed when read: a
    level's x2 downsampling without its input whole. x and source: tensors or Rows (a source that is also x's own input follows x
    by a few rows: give it `retain`)."""

    def __init__(self, x, conv, source=None, shortcut=None):
        super().__init__((x.shape[0], conv.out_channels, x.shape[2] // 2, x.shape[3] // 2), x)
        self.x, self.conv, self.source, self.shortcut = x, conv, source, shortcut

    def compute(self, band, offset, first, last):
        stop = min(self.x.shape[2], 2 * last + 1)
        target = band[:, :, offset:offset + last - first]
        target.copy_(self.conv(F.pad(_read(self.x, 2 * first, stop), (0, 1, 0, 2 * last + 1 - stop))))
        if self.source is not None:
            target.add_(self.shortcut(_read(self.source, 2 * first, 2 * last)))


class NormConvRows(Rows):
    """conv(norm_act(x)) (pixel_norm_conv's arguments) on x, a tensor or Rows, as rows computed when read."""

    def __init__(self, x, norm_act, conv):
        super().__init__((x.shape[0], conv.out_channels, *x.shape[2:]), x)
        self.x, self.norm_act, self.conv = x, norm_act, conv

    def compute(self, band, offset, first, last):
        lo = max(0, first - 1)
        self.conv(self.norm_act(_read(self.x, lo, min(self.shape[2], last + 1))), window=(first - lo, last - first), out=band, out_row=offset)


def chain_levels(levels, output, batch, itemsize, band_bytes=BAND_BYTES):
    """Which levels of a decoder to compute as a chain of rows: (first level, rows per band of the output); no level: len(levels).
    levels[k] = (height, width, channels) of the x2 activation that level k reads, None for a level that reads none or that cannot
    be chained; output = (height, width, channels of its widest full-size activation). A level joins from the first one where both
    hold: that activation takes more than CHAIN_MIN_BANDS bands and its share of an output band keeps the size from which the
    row-window convolution is used (CHAIN_MIN_PIXELS). Larger levels follow."""
    height, width, widest = output
    count = band_rows(height, batch * widest * width * itemsize, 0, band_bytes)
    for level, size in enumerate(levels):
        if size is not None and batch * size[2] * size[0] * size[1] * itemsize > CHAIN_MIN_BANDS * band_bytes and (count * size[0] // height) * size[1] >= CHAIN_MIN_PIXELS:
            return level, count
    return len(levels), count


def bands(rows, count):
    """(first, last, band) down the image for rows (Rows), `count` rows at a time: each band is valid until the next one is read."""
    for start in range(0, rows.shape[2], count):
        stop = min(rows.shape[2], start + count)
        yield start, stop, rows.rows(start, stop)


def collect(rows, count):
    """The whole activation of rows (Rows), read `count` rows at a time."""
    n, c, height, width = rows.shape
    out = torch.empty((n, c, height, width), dtype=rows.dtype, device=rows.device, memory_format=torch.channels_last if rows.channels_last else torch.contiguous_format)
    for start, stop, band in bands(rows, count):
        out[:, :, start:stop].copy_(band)
    return out
