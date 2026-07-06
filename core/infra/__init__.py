"""Infrastructure: settings, templates, logging, debug. Layer 2."""
from core.infra.settings import (
    load_settings, save_settings,
    load_routes, save_routes,
    load_chains, save_chains,
    load_plugins, save_plugins,
    get_route, get_agent_workspace, collect_done_tasks,
)
from core.infra.templates import (
    THINKING_PRESETS, THINKING_BUDGET_REVERSE,
    normalize_thinking, resolve_variable, resolve_template,
)
from core.infra.logger import (
    trace_log, log_model_req, trace_model,
    log_request_start, log_round, log_request_end,
    dump_owui, query_logs, get_log_detail, get_requests_path,
)
from core.infra.debug import DebugContext, is_debug_enabled, notify_activity
