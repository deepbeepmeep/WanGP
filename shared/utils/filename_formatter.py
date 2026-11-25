# Copyright (c) 2025-2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""
Filename formatter for customizing output filenames using template syntax.

Example usage:
    from shared.utils.filename_formatter import FilenameFormatter

    template = "{date}-{prompt(50)}-{seed}"
    settings = {"prompt": "A beautiful sunset over the ocean", "seed": 12345}
    filename = FilenameFormatter.format_filename(template, settings)
    # Result: "2025-01-15-14h30m45s-A_beautiful_sunset_over_the_ocean-12345"

    Placeholder syntax:
        {key}                -> value from the settings
        {key(arg)}           -> optional argument (date format, or max length for {prompt})
        {key:fmt}            -> Python format spec applied to the value, e.g. {scale:.2f}, {steps:.0f}

Date format examples:
    {date}                      -> 2025-01-15-14h30m45s (default)
    {date(YYYY-MM-DD)}          -> 2025-01-15
    {date(YYYY/MM/DD)}          -> 2025/01/15
    {date(DD.MM.YYYY)}          -> 15.01.2025
    {date(YYYY-MM-DD_HH-mm-ss)} -> 2025-01-15_14-30-45
    {date(HHhmm)}               -> 14h30
"""

import re
import time
from datetime import datetime


class FilenameFormatter:
    """
    Formats output filenames using template syntax with settings values.

    Supported placeholders:
    - {date} - timestamp with default format YYYY-MM-DD-HHhmmss
    - {date(YYYY-MM-DD)} - date with custom format and separator
    - {date(YYYY-MM-DD_HH-mm-ss)} - date and time with custom separators
    - {seed} - generation seed
    - {model} or {model_type} - model type (e.g., "ltx2.5", "wan1.3")
    - {resolution} - selected resolution (e.g., "1280x720 (16:9)")
    - {width} / {height} - effective output dimensions (after spatial upsampling)
    - {fps} - effective output fps (after temporal upsampling)
    - {scale} - spatial upsampling factor (1 when no spatial upsampler is used)
    - {num_inference_steps} or {steps} - number of inference steps
    - {prompt} or {prompt(50)} - prompt text with optional max length
    - {flow_shift} - flow shift value
    - {video_length} or {frames} - output frame count
    - {guidance_scale} or {cfg} - guidance scale value
    - {teacache} - TeaCache tag (e.g., "tc1.00"), empty when TeaCache is disabled
    - {sliding_window_size} / {sliding_window_overlap} - sliding window size/overlap in frames
    - {temporal_upsampling} / {spatial_upsampling} - selected upsampler (empty when unused)

    A Python format spec can be appended to any placeholder: {key:fmt}
    (e.g., {scale:.2f}, {steps:.0f}, {fps:.0f}).

    Date format tokens:
    - YYYY: 4-digit year (2025)
    - YY: 2-digit year (25)
    - MM: 2-digit month (01-12)
    - DD: 2-digit day (01-31)
    - HH: 2-digit hour 24h (00-23)
    - hh: 2-digit hour 12h (01-12)
    - mm: 2-digit minute (00-59)
    - ss: 2-digit second (00-59)
    - Separators: - _ . : / and space

    Example templates:
    - "{date}-{prompt(50)}-{seed}"
    - "{date(YYYYMMDD)}-{resolution}-{steps}steps"
    - "{date(YYYY-MM-DD_HH-mm-ss)}_{seed}"
    """

    # Allowed placeholder keys (with aliases)
    ALLOWED_KEYS = {
        'date', 'seed', 'resolution', 'num_inference_steps', 'steps',
        'prompt', 'flow_shift', 'video_length', 'frames',
        'guidance_scale', 'cfg',
        'model', 'model_type', 'fps', 'scale', 'width', 'height',
        'teacache', 'sliding_window_size', 'sliding_window_overlap',
        'temporal_upsampling', 'spatial_upsampling'
    }

    # Map aliases to actual setting keys
    KEY_ALIASES = {
        'steps': 'num_inference_steps',
        'frames': 'video_length',
        'cfg': 'guidance_scale',
        'model': 'model_type'
    }

    # Pattern to match placeholders: {key}, {key(arg)}, {key:fmt}, or {key(arg):fmt}
    PLACEHOLDER_PATTERN = re.compile(r'\{(\w+)(?:\(([^)]*)\))?(:([^}]*))?\}')

    # Date token to strftime mapping (order matters - longer tokens first)
    DATE_TOKENS = [
        ('YYYY', '%Y'),  # 4-digit year
        ('YY', '%y'),    # 2-digit year
        ('MM', '%m'),    # 2-digit month
        ('DD', '%d'),    # 2-digit day
        ('HH', '%H'),    # 2-digit hour (24h)
        ('hh', '%I'),    # 2-digit hour (12h)
        ('mm', '%M'),    # 2-digit minute
        ('ss', '%S'),    # 2-digit second
    ]

    # Allowed separator characters in date format
    DATE_SEPARATORS = set('-_.:/ h')

    # Characters not allowed in filenames (covers Windows, macOS, Linux)
    UNSAFE_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f\n\r\t/]')

    def __init__(self, template: str):
        """
        Initialize with a template string.

        Args:
            template: Format string like "{date}-{prompt(50)}-{seed}"

        Raises:
            ValueError: If template contains unknown placeholders
        """
        self.template = template
        self._validate_template()

    def _validate_template(self):
        """Validate that template only uses allowed placeholders."""
        for match in self.PLACEHOLDER_PATTERN.finditer(self.template):
            key = match.group(1)
            if key not in self.ALLOWED_KEYS:
                allowed = ', '.join(sorted(self.ALLOWED_KEYS))
                raise ValueError(f"Unknown placeholder: {{{key}}}. Allowed: {allowed}")

    def _parse_date_format(self, fmt: str) -> str:
        """
        Convert user-friendly date format to strftime format.

        Args:
            fmt: User format like "YYYY-MM-DD" or "YYYY/MM/DD_HH-mm-ss"

        Returns:
            strftime format string like "%Y-%m-%d" or "%Y/%m/%d_%H-%M-%S"
        """
        result = fmt

        # Replace tokens with strftime codes (longer tokens first to avoid partial matches)
        for token, strftime_code in self.DATE_TOKENS:
            result = result.replace(token, strftime_code)

        return result

    def _is_valid_date_format(self, fmt: str) -> bool:
        """
        Check if date format string contains only valid tokens and separators.

        Args:
            fmt: User format string to validate

        Returns:
            True if format is valid and safe
        """
        # Make a copy to check
        remaining = fmt

        # Remove all valid tokens
        for token, _ in self.DATE_TOKENS:
            remaining = remaining.replace(token, '')

        # What's left should only be separators
        return all(c in self.DATE_SEPARATORS for c in remaining)

    def _format_date(self, arg: str = None) -> str:
        """
        Format current timestamp.

        Args:
            arg: Optional date format string like "YYYY-MM-DD" or "HH:mm:ss"
                 If None or invalid, uses default format.

        Returns:
            Formatted date/time string
        """
        default_fmt = "%Y-%m-%d-%Hh%Mm%Ss"

        if arg is None:
            strftime_fmt = default_fmt
        elif self._is_valid_date_format(arg):
            strftime_fmt = self._parse_date_format(arg)
        else:
            # Invalid format, use default
            strftime_fmt = default_fmt

        try:
            return datetime.fromtimestamp(time.time()).strftime(strftime_fmt)
        except Exception:
            return datetime.fromtimestamp(time.time()).strftime(default_fmt)

    def _truncate(self, value: str, max_len: int) -> str:
        """Truncate string to max length."""
        if max_len <= 0 or len(value) <= max_len:
            return value
        return value[:max_len].rstrip()

    def _sanitize_for_filename(self, value: str) -> str:
        """
        Remove/replace characters unsafe for filenames.

        - Replaces unsafe chars with underscore
        - Collapses multiple underscores/spaces
        - Strips leading/trailing underscores and spaces
        """
        if not value:
            return ''

        # Replace unsafe chars with underscore
        sanitized = self.UNSAFE_FILENAME_CHARS.sub('_', str(value))

        # Replace multiple underscores/spaces with single underscore
        sanitized = re.sub(r'[_\s]+', '_', sanitized)

        # Strip leading/trailing underscores and spaces
        return sanitized.strip('_ ')

    def format(self, settings: dict) -> str:
        """
        Format the template with settings values.

        Args:
            settings: Dictionary containing settings values

        Returns:
            Formatted filename (without extension), safe for filesystem
        """
        def replace_placeholder(match):
            key = match.group(1)
            arg = match.group(2)  # Optional argument in parentheses
            fmt_spec = match.group(4)  # Optional Python format spec (e.g. ".2f"), None when absent

            # Handle date specially
            if key == 'date':
                return self._format_date(arg)

            # Teacache: composed of the skip-steps cache type and its multiplier
            # (e.g. "tc1.00"); empty when the cache is disabled
            if key == 'teacache':
                cache_type = settings.get('skip_steps_cache_type')
                if not cache_type:
                    return ''
                multiplier = settings.get('skip_steps_multiplier')
                try:
                    return self._sanitize_for_filename(f"{cache_type}{float(multiplier):.2f}")
                except (TypeError, ValueError):
                    return self._sanitize_for_filename(str(cache_type))

            # Resolve aliases
            actual_key = self.KEY_ALIASES.get(key, key)

            # Get value from settings
            value = settings.get(actual_key)

            # Apply Python format spec if requested (e.g. {scale:.2f})
            if fmt_spec is not None:
                try:
                    if value is None:
                        value = ''
                    elif isinstance(value, (int, float)) and not isinstance(value, bool):
                        value = format(value, fmt_spec)
                    else:
                        value = format(str(value), fmt_spec)
                except (ValueError, TypeError):
                    value = '' if value is None else str(value)
                return self._sanitize_for_filename(value)

            # Convert to string
            if value is None:
                value = ''
            else:
                value = str(value)

            # Apply truncation if specified (for text fields)
            if arg is not None and arg.isdigit():
                max_len = int(arg)
                value = self._truncate(value, max_len)

            return self._sanitize_for_filename(value)

        result = self.PLACEHOLDER_PATTERN.sub(replace_placeholder, self.template)

        # Sanitize any literal text in template that might be unsafe
        result = self._sanitize_for_filename(result)

        # Ensure result is not empty
        if not result:
            result = self._format_date()

        return result

    @classmethod
    def format_filename(cls, template: str, settings: dict) -> str:
        """
        Convenience class method to format a filename in one call.

        Args:
            template: Format string like "{date}-{prompt(50)}-{seed}"
            settings: Dictionary containing settings values

        Returns:
            Formatted filename (without extension)

        Raises:
            ValueError: If template contains unknown placeholders
        """
        formatter = cls(template)
        return formatter.format(settings)

    @classmethod
    def get_help_text(cls) -> str:
        """Return help text describing the template syntax."""
        return """Filename Template Syntax:

Placeholders (wrap in curly braces):
  {date}                  - Timestamp (default: 2025-01-15-14h30m45s)
  {date(YYYY-MM-DD)}      - Date with custom format
  {date(HH-mm-ss)}        - Time only
  {date(YYYY-MM-DD_HH-mm-ss)} - Date and time
  {seed}                  - Generation seed
  {resolution}            - Video resolution (e.g., 1280x720)
  {num_inference_steps}   - Number of inference steps (alias: {steps})
  {prompt}                - Full prompt text
  {prompt(50)}            - Prompt truncated to 50 characters
  {flow_shift}            - Flow shift value
  {video_length}          - Video length in frames (alias: {frames})
  {guidance_scale}        - Guidance scale (alias: {cfg})

Date/Time tokens:
  YYYY - 4-digit year     MM - month (01-12)    DD - day (01-31)
  HH   - hour 24h (00-23) hh - hour 12h (01-12)
  mm   - minute (00-59)   ss - second (00-59)
  Separators: - _ . : / space h

Examples:
  {date}-{prompt(50)}-{seed}
  {date(YYYYMMDD)}_{resolution}_{steps}steps
  {date(YYYY-MM-DD_HH-mm-ss)}_{seed}
  {date(DD.MM.YYYY)}_{prompt(30)}
"""
