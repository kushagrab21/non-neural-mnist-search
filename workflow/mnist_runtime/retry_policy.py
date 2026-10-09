"""Explicit retry eligibility; never infer safe repetition from local timeout."""
POLICY = {'version': 'bounded-retry/1', 'max_retries_per_logical_task': 1,
          'max_extra_tasks': 12, 'backoff_seconds': 5,
          'malformed_structured_output': 'one correction task after fully accounted successful provider completion',
          'transient_codes': [429, 500, 502, 503, 504],
          'never_retry': ['unknown_completion', 'unknown_usage', 'cleanup_failure', 'budget_exhaustion',
                          'model_substitution', 'provider_safeguard', 'scientific_rejection', 'semantic_failure'],
          'evaluation_retries': 0, 'build_retries': 0,
          'accounting': 'Each retry has a new dispatch and reservation; original evidence remains immutable.'}


def eligibility(failure, *, attempts, extras, policy=POLICY):
    if attempts >= 1 or extras >= policy['max_extra_tasks']:
        return False, 'retry allocation exhausted'
    required = ('terminal_receipt', 'usage_known', 'cleanup_confirmed', 'same_session_model', 'within_caps')
    if not all(failure.get(k) is True for k in required):
        return False, 'completion, accounting or identity not established'
    if failure.get('safeguard_flag') or failure.get('budget_exhausted'):
        return False, 'provider safeguard or budget stop'
    if failure.get('kind') == 'malformed_structured_output' and failure.get('provider_success') is True:
        return True, 'one recorded output correction'
    if failure.get('kind') == 'transient_provider_error' and failure.get('http_status') in policy['transient_codes']:
        return True, 'one accounted transient retry'
    return False, 'failure requires review rather than repetition'
